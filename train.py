"""Train SChanger on a paired change-detection dataset.

Expected layout:

    DATA_ROOT/
      train/{A,B,label}/
      val/{A,B,label}/

Folder names and split names can be changed with command-line options.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import albumentations as A
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASET_SUFFIXES = {
    "levir-cd": "levir",
    "levir-cd+": "levir_plus",
    "s2looking": "s2looking",
    "cdd": "cdd",
    "sysu-cd": "sysu",
    "whu-cd": "whu",
}
MODEL_CHANNELS = {
    "small": [8, 16, 32, 40, 48, 48],
    "base": [24, 32, 48, 64, 104, 120],
}
PRESET_FIELDS = (
    "image_size",
    "batch_size",
    "epochs",
    "workers",
    "prefetch_factor",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "noise_probability",
    "swap_probability",
    "ema_decay",
    "ema_steps",
    "eval_interval",
    "amp",
    "ema",
    "seed",
)


def apply_training_preset(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    args.init_url = None
    try:
        document = json.loads(args.config.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"training configuration not found: {args.config}")
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read training configuration {args.config}: {error}")

    try:
        preset = document["datasets"][args.dataset]
    except (KeyError, TypeError):
        parser.error(f"configuration {args.config} has no preset for {args.dataset}")
    if not isinstance(preset, dict):
        parser.error(f"preset for {args.dataset} must be a JSON object")

    missing = [name for name in PRESET_FIELDS if name not in preset]
    if missing:
        parser.error(f"preset for {args.dataset} is missing: {', '.join(missing)}")
    for name in PRESET_FIELDS:
        if getattr(args, name) is None:
            setattr(args, name, preset[name])

    if args.init is None and args.resume is None and not args.from_scratch:
        try:
            configured_weight = document["pretrained_weights"][args.model]
        except (KeyError, TypeError):
            parser.error(f"configuration {args.config} has no {args.model} pretrained weight")
        if not isinstance(configured_weight, dict):
            parser.error(f"pretrained weight for {args.model} must be a JSON object")
        try:
            args.init = Path(configured_weight["path"])
            args.init_url = configured_weight["url"]
        except (KeyError, TypeError):
            parser.error(f"pretrained weight for {args.model} requires path and url")
        if not args.init.is_absolute():
            args.init = PROJECT_ROOT / args.init
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_SUFFIXES))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_CHANNELS), default="base")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training_configs.json",
        help="Dataset training presets (default: training_configs.json beside this script)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--image-dirs", nargs=2, default=("A", "B"), metavar=("FIRST", "SECOND"))
    parser.add_argument("--label-dir", default="label")
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--compute-normalization", action="store_true")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--noise-probability", type=float)
    parser.add_argument("--swap-probability", type=float)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--ema-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=10)
    parser.add_argument("--init", type=Path, help="Initialize model weights without optimizer state")
    parser.add_argument("--resume", type=Path, help="Resume a checkpoint created by this script")
    parser.add_argument("--from-scratch", action="store_true", help="Disable pretrained initialization")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    args = apply_training_preset(parser, parser.parse_args(argv))

    if args.batch_size < 1 or args.epochs < 1 or args.image_size < 16:
        parser.error("batch-size and epochs must be positive; image-size must be at least 16")
    if args.workers < 0 or args.prefetch_factor < 1 or args.warmup_epochs < 0:
        parser.error("workers and warmup-epochs must be non-negative; prefetch-factor must be positive")
    if args.warmup_epochs >= args.epochs:
        parser.error("warmup-epochs must be smaller than epochs")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr must be positive and weight-decay must be non-negative")
    for name in ("noise_probability", "swap_probability"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(name.replace("_", "-") + " must be in [0, 1]")
    if not 0 < args.ema_decay < 1 or args.ema_steps < 1:
        parser.error("ema-decay must be in (0, 1) and ema-steps must be positive")
    if args.eval_interval < 1 or args.save_interval < 1 or args.keep_checkpoints < 1:
        parser.error("eval-interval, save-interval, and keep-checkpoints must be positive")
    if sum(value is not None and value is not False for value in (args.init, args.resume, args.from_scratch)) > 1:
        parser.error("--init, --resume, and --from-scratch are mutually exclusive")
    folders = (*args.image_dirs, args.label_dir)
    if len(set(folders)) != 3 or any(Path(folder).name != folder for folder in folders):
        parser.error("image-dirs and label-dir must be three distinct folder names")

    suffix = DATASET_SUFFIXES[args.dataset]
    args.output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "training" / f"{suffix}-{args.model}"
    return args


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"No supported images found in {directory}")
    return result


def paired_samples(root: Path, groups: tuple[str, str, str]) -> list[tuple[Path, Path, Path]]:
    mappings = [files_by_stem(root / group) for group in groups]
    stems = set(mappings[0])
    for group, mapping in zip(groups[1:], mappings[1:]):
        if set(mapping) != stems:
            missing = sorted(stems - set(mapping))[:5]
            extra = sorted(set(mapping) - stems)[:5]
            raise ValueError(f"Unpaired files in {root / group}: missing={missing}, extra={extra}")
    return [(mappings[0][stem], mappings[1][stem], mappings[2][stem]) for stem in sorted(stems)]


class ChangeDataset(Dataset):
    def __init__(
        self,
        root: Path,
        image_dirs: tuple[str, str],
        label_dir: str,
        normalization: dict,
        train: bool,
        image_size: int,
        noise_probability: float,
        swap_probability: float,
    ) -> None:
        self.samples = paired_samples(root, (*image_dirs, label_dir))
        self.train = train
        self.image_size = image_size
        self.swap_probability = swap_probability
        self.geometric_transforms = A.Compose([
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.Rotate(limit=45, p=0.3),
            A.ShiftScaleRotate(p=0.3),
        ], additional_targets={"image1": "image"})
        self.image_transforms = A.Compose([A.OneOf([
            A.GaussNoise(p=1.0),
            A.HueSaturationValue(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0),
            A.Emboss(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=noise_probability)], additional_targets={"image1": "image"})
        self.means = [np.asarray(normalization[key]["means"], dtype=np.float32) for key in ("A", "B")]
        self.stds = [np.asarray(normalization[key]["stds"], dtype=np.float32) for key in ("A", "B")]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_path, second_path, label_path = self.samples[index]
        with Image.open(first_path) as image:
            first = image.convert("RGB")
        with Image.open(second_path) as image:
            second = image.convert("RGB")
        with Image.open(label_path) as image:
            label = image.convert("L")
        if first.size != second.size or first.size != label.size:
            raise ValueError(f"Image/label sizes differ for {first_path.stem}")

        if self.train:
            sample = self.geometric_transforms(
                image=np.asarray(first), image1=np.asarray(second), mask=np.asarray(label))
            label = sample["mask"]
            images = self.image_transforms(image=sample["image"], image1=sample["image1"])
            first, second = images["image"], images["image1"]

        arrays = [np.asarray(image, dtype=np.float32) / 255.0 for image in (first, second)]
        tensors = []
        for pixels, mean, std in zip(arrays, self.means, self.stds):
            normalized = (pixels - mean) / std
            tensors.append(torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))))
        if self.train and random.random() < self.swap_probability:
            tensors.reverse()
        target = torch.from_numpy(np.ascontiguousarray(np.asarray(label) != 0)).float()
        return tensors[0], tensors[1], target


def validate_normalization(value: dict) -> dict:
    for key in ("A", "B"):
        if key not in value or "means" not in value[key] or "stds" not in value[key]:
            raise ValueError(f"Normalization must contain {key}.means and {key}.stds")
        means = np.asarray(value[key]["means"], dtype=np.float64)
        stds = np.asarray(value[key]["stds"], dtype=np.float64)
        if means.shape != (3,) or stds.shape != (3,) or not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise ValueError(f"Normalization {key} must contain three finite RGB means and stds")
        if (means < 0).any() or (means > 1).any() or (stds <= 0).any():
            raise ValueError(f"Normalization {key} means must be in [0, 1] and stds must be positive")
    return value


def compute_normalization(root: Path, image_dirs: tuple[str, str]) -> dict:
    result: dict[str, dict] = {}
    for key, folder in zip(("A", "B"), image_dirs):
        paths = files_by_stem(root / folder)
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for index, path in enumerate(paths.values(), start=1):
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            means.append(pixels.mean(axis=(0, 1)))
            stds.append(pixels.std(axis=(0, 1)))
            if index % 100 == 0 or index == len(paths):
                print(f"Normalization {key}: {index}/{len(paths)}", flush=True)
        result[key] = {"means": np.mean(means, axis=0).tolist(),
                       "stds": np.mean(stds, axis=0).tolist(), "images": len(paths)}
    result["method"] = "mean of per-image RGB mean/std on training data, scaled to [0,1]"
    return validate_normalization(result)


class DiceBCELoss(nn.Module):
    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError(f"Unexpected logits/target shapes: {tuple(logits.shape)}, {tuple(target.shape)}")
        expanded_target = target[:, None].expand_as(logits)
        probabilities = torch.sigmoid(logits)
        dimensions = (0, 2, 3)
        intersection = (probabilities * expanded_target).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + expanded_target.sum(dim=dimensions)
        dice = 1.0 - (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        bce = torch.stack([self.bce(logits[:, channel], target) for channel in range(logits.shape[1])]).sum()
        return bce + dice.sum()


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    groups = [{"params": [], "weight_decay": 0.0}, {"params": [], "weight_decay": weight_decay}]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[1 if use_decay else 0]["params"].append(parameter)
    return groups


def cosine_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    def multiplier(step: int) -> float:
        if warmup_steps and step <= warmup_steps:
            alpha = step / warmup_steps
            return 1e-3 * (1.0 - alpha) + alpha
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 1e-6 + 0.5 * (1.0 - 1e-6) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float, device: torch.device) -> None:
        self.module = copy.deepcopy(model).eval().to(device)
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, destination in self.module.state_dict().items():
            value = source[name].detach().to(destination.device)
            if destination.is_floating_point():
                destination.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                destination.copy_(value)


def checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Expected a checkpoint containing 'state_dict'")
    state = payload["state_dict"]
    if not state or not all(isinstance(name, str) and isinstance(value, torch.Tensor)
                            for name, value in state.items()):
        raise ValueError("Checkpoint 'state_dict' must contain named tensors")
    return state


def load_checkpoint_file(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


def ensure_initial_weights(path: Path, url: str | None) -> None:
    if path.is_file():
        return
    if url is None:
        raise FileNotFoundError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    temporary.unlink(missing_ok=True)
    print(f"Downloading pretrained weights to {path}", flush=True)
    try:
        torch.hub.download_url_to_file(url, temporary, progress=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_initial_weights(
    model: nn.Module,
    path: Path,
    url: str | None = None,
) -> None:
    ensure_initial_weights(path, url)
    payload = load_checkpoint_file(path)
    missing, unexpected = model.load_state_dict(checkpoint_state(payload), strict=False)
    print(f"Initialized from {path}; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if missing:
        print("Missing keys:", *missing, sep="\n  ")
    if unexpected:
        print("Unexpected keys:", *unexpected, sep="\n  ")


def confusion_counts(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = probabilities > 0.5
    truth = target.bool()
    return torch.bincount((truth.long() * 2 + prediction.long()).flatten(), minlength=4)


def f1_from_counts(counts: torch.Tensor) -> float:
    _, fp, fn, tp = [int(value) for value in counts]
    denominator = 2 * tp + fp + fn
    return 100.0 * 2 * tp / denominator if denominator else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    ema: ModelEMA | None,
    ema_steps: int,
    global_step: int,
) -> tuple[dict, int]:
    model.train()
    amp_enabled = scaler.is_enabled()
    total_loss = 0.0
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    samples = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        if len(batch) == 3:
            first, second, target = batch
            model_input = (first.to(device, non_blocking=True), second.to(device, non_blocking=True))
        elif len(batch) == 2:
            first, target = batch
            model_input = first.to(device, non_blocking=True)
        else:
            raise ValueError(f"Expected a two- or three-tensor training batch, got {len(batch)} items")
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(model_input)
            loss = criterion(logits.float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {global_step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema is not None and global_step % ema_steps == 0:
            ema.update(model)
        scheduler.step()
        global_step += 1
        batch_size = len(target)
        total_loss += loss.item() * batch_size
        samples += batch_size
        counts += confusion_counts(torch.sigmoid(logits[:, 0]), target)
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"  batch {batch_index}/{len(loader)} loss={total_loss / samples:.5f}", flush=True)
    return {"loss": total_loss / samples, "f1": f1_from_counts(counts),
            "seconds": time.perf_counter() - start}, global_step


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    samples = 0
    start = time.perf_counter()
    for batch in loader:
        if len(batch) == 3:
            first, second, target = batch
            model_input = (first.to(device, non_blocking=True), second.to(device, non_blocking=True))
        elif len(batch) == 2:
            first, target = batch
            model_input = first.to(device, non_blocking=True)
        else:
            raise ValueError(f"Expected a two- or three-tensor validation batch, got {len(batch)} items")
        target = target.to(device, non_blocking=True)
        probabilities = model(model_input)[:, 0]
        counts += confusion_counts(probabilities, target)
        samples += len(target)
    return {"f1": f1_from_counts(counts), "samples": samples,
            "confusion": [int(value) for value in counts], "seconds": time.perf_counter() - start}


def serializable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def prune_epoch_checkpoints(directory: Path, keep: int) -> None:
    checkpoints = []
    for path in directory.glob("epoch-*.pth"):
        try:
            epoch = int(path.stem.removeprefix("epoch-"))
        except ValueError:
            continue
        checkpoints.append((epoch, path))
    for _, path in sorted(checkpoints)[:-keep]:
        path.unlink()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    sys.path.insert(0, str(PROJECT_ROOT))
    from cd.SChanger import SChanger

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = args.normalization
    if normalization_path is None and not args.compute_normalization:
        normalization_path = PROJECT_ROOT / "evaluation" / "normalization" / f"{DATASET_SUFFIXES[args.dataset]}.json"
    if args.compute_normalization:
        normalization = compute_normalization(args.data_root / args.train_split, tuple(args.image_dirs))
    else:
        normalization = validate_normalization(json.loads(normalization_path.read_text(encoding="utf-8")))
    (args.output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")

    train_dataset = ChangeDataset(args.data_root / args.train_split, tuple(args.image_dirs), args.label_dir,
                                  normalization, True, args.image_size, args.noise_probability, args.swap_probability)
    validation_dataset = None
    if args.validate:
        validation_dataset = ChangeDataset(args.data_root / args.val_split, tuple(args.image_dirs), args.label_dir,
                                           normalization, False, args.image_size, 0.0, 0.0)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda",
                      "worker_init_fn": seed_worker, "generator": generator}
    if args.workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_options)
    if not len(train_loader):
        raise ValueError("Training split is smaller than one batch")
    validation_loader = (DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                    drop_last=False, **loader_options) if validation_dataset else None)

    model = SChanger(num_classes=1, input_channels=3, c_list=MODEL_CHANNELS[args.model], dropout=0.2).to(device)
    if args.init:
        load_initial_weights(model, args.init, args.init_url)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    scheduler = cosine_scheduler(optimizer, len(train_loader), args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay, device) if args.ema else None
    criterion = DiceBCELoss()

    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    if args.resume:
        checkpoint = load_checkpoint_file(args.resume)
        model.load_state_dict(checkpoint_state(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None:
            ema_state = checkpoint["ema_state_dict"]
            if not isinstance(ema_state, dict):
                raise ValueError(f"Checkpoint has no EMA state: {args.resume}")
            ema.module.load_state_dict(ema_state, strict=True)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_f1 = float(checkpoint["best_f1"])
        print(f"Resumed {args.resume} at epoch {start_epoch}", flush=True)

    print(f"Device: {device}; train pairs: {len(train_dataset)}; "
          f"validation pairs: {len(validation_dataset) if validation_dataset else 0}", flush=True)
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)
        train_metrics, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                                                     device, scaler, ema, args.ema_steps, global_step)
        record = {"epoch": epoch, "global_step": global_step, "train": train_metrics,
                  "lr": max(group["lr"] for group in optimizer.param_groups)}
        should_validate = validation_loader is not None and ((epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs)
        if should_validate:
            evaluation_model = ema.module if ema is not None else model
            validation_metrics = validate(evaluation_model, validation_loader, device)
            record["validation"] = validation_metrics
            print(f"  validation f1={validation_metrics['f1']:.4f}", flush=True)
            if validation_metrics["f1"] >= best_f1:
                best_f1 = validation_metrics["f1"]
                best_payload = {"state_dict": cpu_state_dict(evaluation_model), "epoch": epoch,
                                "global_step": global_step, "best_f1": best_f1,
                                "normalization": normalization, "args": serializable_args(args)}
                save_checkpoint(args.output_dir / "best.pth", best_payload)

        checkpoint = {"state_dict": cpu_state_dict(model),
                      "ema_state_dict": cpu_state_dict(ema.module) if ema else None,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
                      "best_f1": best_f1, "normalization": normalization, "args": serializable_args(args)}
        save_checkpoint(args.output_dir / "last.pth", checkpoint)
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(args.output_dir / f"epoch-{epoch + 1}.pth", checkpoint)
            prune_epoch_checkpoints(args.output_dir, args.keep_checkpoints)
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"  train loss={train_metrics['loss']:.5f} f1={train_metrics['f1']:.4f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
