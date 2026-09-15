"""Pretrain SChanger on a single-image building segmentation dataset.

Expected layout:

    DATA_ROOT/
      train/images/
      train/labels/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from cd.SChanger import SChanger
from train import (
    DiceBCELoss,
    MODEL_CHANNELS,
    ModelEMA,
    checkpoint_state,
    cosine_scheduler,
    cpu_state_dict,
    files_by_stem,
    load_checkpoint_file,
    parameter_groups,
    prune_epoch_checkpoints,
    save_checkpoint,
    seed_everything,
    seed_worker,
    serializable_args,
    train_one_epoch,
)


PRESET_FIELDS = (
    "image_size", "batch_size", "epochs", "workers", "prefetch_factor", "lr",
    "weight_decay", "warmup_epochs", "noise_probability",
    "color_jitter_probability", "ema", "ema_steps", "ema_decay", "amp",
    "save_interval", "keep_checkpoints", "seed",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", choices=tuple(MODEL_CHANNELS), default="base")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training_configs.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--label-dir", default="labels")
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--noise-probability", type=float)
    parser.add_argument("--color-jitter-probability", type=float)
    parser.add_argument("--ema-steps", type=int)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--keep-checkpoints", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    try:
        document = json.loads(args.config.read_text(encoding="utf-8"))
        preset = document["pretraining"]
    except FileNotFoundError:
        parser.error(f"training configuration not found: {args.config}")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        parser.error(f"cannot read pretraining configuration {args.config}: {error}")
    missing = [name for name in PRESET_FIELDS if name not in preset]
    if missing:
        parser.error(f"pretraining configuration is missing: {', '.join(missing)}")
    for name in PRESET_FIELDS:
        if getattr(args, name) is None:
            setattr(args, name, preset[name])

    if args.batch_size < 1 or args.epochs < 1 or args.image_size < 16:
        parser.error("batch-size and epochs must be positive; image-size must be at least 16")
    if args.workers < 0 or args.prefetch_factor < 1 or args.warmup_epochs < 0:
        parser.error("workers and warmup-epochs must be non-negative; prefetch-factor must be positive")
    if args.warmup_epochs >= args.epochs:
        parser.error("warmup-epochs must be smaller than epochs")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr must be positive and weight-decay must be non-negative")
    for name in ("noise_probability", "color_jitter_probability"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(name.replace("_", "-") + " must be in [0, 1]")
    if not 0 < args.ema_decay < 1 or args.ema_steps < 1:
        parser.error("ema-decay must be in (0, 1) and ema-steps must be positive")
    if args.save_interval < 1 or args.keep_checkpoints < 1:
        parser.error("save-interval and keep-checkpoints must be positive")
    if len({args.image_dir, args.label_dir}) != 2:
        parser.error("image-dir and label-dir must be distinct")
    for folder in (args.image_dir, args.label_dir):
        if Path(folder).name != folder:
            parser.error("image-dir and label-dir must be folder names, not paths")
    args.output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "pretraining" / args.model
    return args


def paired_samples(root: Path, image_dir: str, label_dir: str) -> list[tuple[Path, Path]]:
    images = files_by_stem(root / image_dir)
    labels = files_by_stem(root / label_dir)
    if set(images) != set(labels):
        missing = sorted(set(images) - set(labels))[:5]
        extra = sorted(set(labels) - set(images))[:5]
        raise ValueError(f"Unpaired labels in {root / label_dir}: missing={missing}, extra={extra}")
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def validate_normalization(value: dict) -> dict:
    means = np.asarray(value.get("means"), dtype=np.float64)
    stds = np.asarray(value.get("stds"), dtype=np.float64)
    if means.shape != (3,) or stds.shape != (3,) or not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Normalization must contain three finite RGB means and stds")
    if (means < 0).any() or (means > 1).any() or (stds <= 0).any():
        raise ValueError("Normalization means must be in [0, 1] and stds must be positive")
    return value


def compute_normalization(directory: Path) -> dict:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    paths = list(files_by_stem(directory).values())
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        means.append(pixels.mean(axis=(0, 1)))
        stds.append(pixels.std(axis=(0, 1)))
        if index % 100 == 0 or index == len(paths):
            print(f"Normalization: {index}/{len(paths)}", flush=True)
    result = {
        "means": np.mean(means, axis=0).tolist(),
        "stds": np.mean(stds, axis=0).tolist(),
        "images": len(paths),
        "method": "mean of per-image RGB mean/std on training data, scaled to [0,1]",
    }
    return validate_normalization(result)


class BuildingDataset(Dataset):
    def __init__(self, root: Path, image_dir: str, label_dir: str, normalization: dict,
                 image_size: int, noise_probability: float, color_jitter_probability: float) -> None:
        self.samples = paired_samples(root, image_dir, label_dir)
        self.transforms = A.Compose([
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.Rotate(limit=45, p=0.3),
            A.ShiftScaleRotate(p=0.3),
            A.OneOf([A.GaussNoise(p=1.0), A.RandomGamma(p=1.0), A.Emboss(p=1.0), A.MotionBlur(p=1.0)],
                    p=noise_probability),
            A.ColorJitter(p=color_jitter_probability),
            A.Normalize(mean=normalization["means"], std=normalization["stds"]),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        with Image.open(label_path) as opened:
            label = np.asarray(opened.convert("L"))
        if image.shape[:2] != label.shape:
            raise ValueError(f"Image/label sizes differ for {image_path.stem}")
        transformed = self.transforms(image=image, mask=(label != 0).astype(np.float32))
        return transformed["image"].contiguous(), transformed["mask"].contiguous()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_root = args.data_root / args.train_split
    if args.normalization:
        normalization = validate_normalization(json.loads(args.normalization.read_text(encoding="utf-8")))
    else:
        normalization = compute_normalization(train_root / args.image_dir)
    (args.output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")

    dataset = BuildingDataset(train_root, args.image_dir, args.label_dir, normalization, args.image_size,
                              args.noise_probability, args.color_jitter_probability)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda",
                      "worker_init_fn": seed_worker, "generator": generator}
    if args.workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_options)
    if not len(loader):
        raise ValueError("Training split is smaller than one batch")

    model = SChanger(num_classes=1, input_channels=3, c_list=MODEL_CHANNELS[args.model], dropout=0.2,
                     single_stream=True).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.weight_decay), lr=args.lr)
    scheduler = cosine_scheduler(optimizer, len(loader), args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model, args.ema_decay, device) if args.ema else None
    criterion = DiceBCELoss()

    start_epoch = 0
    global_step = 0
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
        print(f"Resumed {args.resume} at epoch {start_epoch}", flush=True)

    print(f"Device: {device}; model: {args.model}; training images: {len(dataset)}", flush=True)
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        metrics, global_step = train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler,
                                               ema, args.ema_steps, global_step)
        record = {"epoch": epoch, "global_step": global_step, "train": metrics,
                  "lr": max(group["lr"] for group in optimizer.param_groups)}
        checkpoint = {"state_dict": cpu_state_dict(model),
                      "ema_state_dict": cpu_state_dict(ema.module) if ema else None,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
                      "normalization": normalization, "args": serializable_args(args)}
        save_checkpoint(args.output_dir / "last.pth", checkpoint)
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(args.output_dir / f"epoch-{epoch + 1}.pth", checkpoint)
            prune_epoch_checkpoints(args.output_dir, args.keep_checkpoints)
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

    final_payload = {"state_dict": cpu_state_dict(model), "epoch": args.epochs - 1,
                     "normalization": normalization, "args": serializable_args(args)}
    pretrained_path = args.output_dir / f"schanger_{args.model}_pretrain.pth"
    save_checkpoint(pretrained_path, final_payload)
    print(f"Pretrained weights: {pretrained_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
