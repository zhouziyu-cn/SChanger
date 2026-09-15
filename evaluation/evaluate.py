"""Evaluate SChanger checkpoints with non-overlapping 256-pixel tiles."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from functools import lru_cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TILE_SIZE = 256
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASETS = {
    "levir-cd": ("LEVIR-CD", "levir", 2048, None),
    "levir-cd+": ("LEVIR-CD+", "levir_plus", 5568, 256),
    "s2looking": ("S2Looking", "s2looking", 16000, 256),
    "cdd": ("CDD", "cdd", 3000, None),
    "sysu-cd": ("SYSU-CD", "sysu", 4000, None),
    "whu-cd": ("WHU-CD", "whu", 2760, None),
}
WEIGHTS_URL = "https://huggingface.co/Zy-Zhou/schanger/resolve/main"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_checkpoint(path: Path, url: str | None = None) -> None:
    if path.is_file():
        return
    if url is None:
        raise ValueError(f"Checkpoint not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    temporary.unlink(missing_ok=True)
    print(f"Downloading checkpoint to {path}", flush=True)
    try:
        torch.hub.download_url_to_file(url, temporary, progress=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def paired_files(root: Path, groups: tuple[str, ...]) -> list[str]:
    collections = []
    for group in groups:
        directory = root / group
        if not directory.is_dir():
            raise ValueError(f"Missing image directory: {directory}")
        names = {p.name for p in directory.iterdir()
                 if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES}
        if not names:
            raise ValueError(f"No supported images found in {directory}")
        collections.append(names)
    for group, names in zip(groups[1:], collections[1:]):
        if names != collections[0]:
            missing = sorted(collections[0] - names)[:5]
            extra = sorted(names - collections[0])[:5]
            raise ValueError(f"Unpaired images in {root / group}: missing={missing}, extra={extra}")
    return sorted(collections[0])


def validate_normalization(stats: dict) -> dict:
    for group in ("A", "B"):
        try:
            mean = np.asarray(stats[group]["means"], dtype=np.float64)
            std = np.asarray(stats[group]["stds"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Normalization must contain {group}.means and {group}.stds") from exc
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(f"Normalization {group}: expected three RGB means and standard deviations")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"Normalization {group}: values must be finite")
        if (mean < 0).any() or (mean > 1).any() or (std <= 0).any() or (std > 1).any():
            raise ValueError(f"Normalization {group}: means must be in [0, 1], stds in (0, 1]")
    return stats


def train_statistics(root: Path, image_dirs: tuple[str, str], tile_size=None) -> dict:
    """Average training-image (or training-tile) RGB means and population stds."""
    names = paired_files(root / "train", image_dirs)
    stats = {}
    for key, group in zip(("A", "B"), image_dirs):
        means, stds = [], []
        for index, name in enumerate(names):
            path = root / "train" / group / name
            with Image.open(path) as image:
                if image.mode != "RGB":
                    raise ValueError(f"Normalization requires RGB training images: {path}")
                pixels = np.asarray(image).astype(np.float32) / 255.0
            height, width = pixels.shape[:2]
            if tile_size and (height % tile_size or width % tile_size):
                raise ValueError(f"Training image dimensions must be multiples of {tile_size}: {path}")
            patches = (pixels[y:y + tile_size, x:x + tile_size]
                       for x in range(0, width, tile_size)
                       for y in range(0, height, tile_size)) if tile_size else (pixels,)
            for patch in patches:
                means.append([float(patch[:, :, channel].mean()) for channel in range(3)])
                stds.append([float(patch[:, :, channel].std()) for channel in range(3)])
            if (index + 1) % 100 == 0:
                print(f"Normalization {group}: {index + 1}/{len(names)}", flush=True)
        stats[key] = {"means": np.mean(means, axis=0).tolist(),
                      "stds": np.mean(stds, axis=0).tolist(), "images": len(names), "samples": len(means)}
    stats["method"] = "mean of per-sample RGB mean/std on training data, scaled to [0,1]"
    stats["tile_size"] = tile_size
    return validate_normalization(stats)


class PairedTiles(Dataset):
    """Paired images, split into complete non-overlapping tiles without resizing."""

    def __init__(self, root: Path, stats: dict, image_dirs: tuple[str, str] = ("A", "B"), label_dir="label"):
        self.root = root
        self.groups = (*image_dirs, label_dir)
        if len(set(self.groups)) != 3:
            raise ValueError("The two image directories and label directory must be distinct")
        self.names = paired_files(root, self.groups)
        validate_normalization(stats)
        self.means = [np.asarray(stats[g]["means"], dtype=np.float32) * 255.0 for g in ("A", "B")]
        self.scales = [1.0 / (np.asarray(stats[g]["stds"], dtype=np.float32) * 255.0) for g in ("A", "B")]
        self.sizes, self.offsets = [], [0]
        for name in self.names:
            sizes = []
            for group in self.groups:
                path = root / group / name
                with Image.open(path) as image:
                    if group != label_dir and image.mode != "RGB":
                        raise ValueError(f"Expected RGB image: {path}, got {image.mode}")
                    if group == label_dir and image.mode not in ("L", "1", "P"):
                        raise ValueError(f"Expected single-channel binary label: {path}")
                    sizes.append(image.size)
            width, height = sizes[0]
            if sizes.count(sizes[0]) != 3:
                raise ValueError(f"Image/label sizes differ for {name}: {sizes}")
            if width % TILE_SIZE or height % TILE_SIZE:
                raise ValueError(f"Image dimensions must be multiples of {TILE_SIZE}: {name} {sizes[0]}")
            self.sizes.append((width, height))
            self.offsets.append(self.offsets[-1] + (width // TILE_SIZE) * (height // TILE_SIZE))
        self.total_pixels = sum(width * height for width, height in self.sizes)

    def __len__(self):
        return self.offsets[-1]

    @lru_cache(maxsize=2)
    def images(self, image_index):
        arrays = []
        for group in self.groups:
            with Image.open(self.root / group / self.names[image_index]) as image:
                arrays.append(np.array(image))
        return arrays

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image_index = bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[image_index]
        rows = self.sizes[image_index][1] // TILE_SIZE
        x, y = (local_index // rows) * TILE_SIZE, (local_index % rows) * TILE_SIZE
        first, second, label = self.images(image_index)
        tensors = []
        for image, mean, scale in zip((first, second), self.means, self.scales):
            pixels = image[y:y + TILE_SIZE, x:x + TILE_SIZE].astype(np.float32)
            pixels = (pixels - mean) * scale
            tensors.append(torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))))
        mask = np.ascontiguousarray(label[y:y + TILE_SIZE, x:x + TILE_SIZE] != 0)
        return tensors[0], tensors[1], torch.from_numpy(mask)


def metrics_from_confusion(counts: list[int]) -> dict:
    tn, fp, fn, tp = counts
    if any(value < 0 for value in counts) or sum(counts) == 0:
        raise ValueError("Confusion counts must be non-negative and include at least one pixel")

    def percent(numerator, denominator):
        return 100.0 * numerator / denominator if denominator else 0.0

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "f1_percent": percent(2 * tp, 2 * tp + fp + fn),
            "precision_percent": percent(tp, tp + fp),
            "recall_percent": percent(tp, tp + fn),
            "iou_percent": percent(tp, tp + fp + fn),
            "accuracy_percent": percent(tp + tn, sum(counts))}


def evaluate(model, loader, device, threshold=0.5, log_interval=32):
    model.to(device).eval()
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    processed = 0
    with torch.inference_mode():
        for batch_index, (first, second, label) in enumerate(loader):
            first, second = first.to(device), second.to(device)
            probabilities = model((first, second))
            if probabilities.shape != (label.shape[0], 1, *label.shape[1:]):
                raise ValueError(f"Unexpected model output shape: {tuple(probabilities.shape)}")
            probabilities = probabilities[:, 0]
            if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
                raise ValueError("Model output must contain finite probabilities in [0, 1]")
            pred = probabilities > threshold
            truth = label.to(device=device, dtype=torch.bool)
            counts += torch.bincount((truth.long() * 2 + pred.long()).flatten(), minlength=4)
            processed += len(label)
            if log_interval and (batch_index + 1) % log_interval == 0:
                print(f"Evaluated {processed}/{len(loader.dataset)} tiles", flush=True)
    return counts.cpu().tolist(), processed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="levir-cd")
    parser.add_argument("--data-root", type=Path, help="Dataset root; defaults to data/<dataset folder>")
    parser.add_argument("--weights-dir", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--model", choices=("base", "small", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path, help="Single local checkpoint; requires --model base or small")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--image-dirs", nargs=2, default=("A", "B"), metavar=("FIRST", "SECOND"))
    parser.add_argument("--label-dir", default="label")
    parser.add_argument("--normalization", type=Path, help="Reuse a saved normalization.json instead of reading train/")
    parser.add_argument("--compute-normalization", action="store_true", help="Compute statistics from train/ instead of bundled values")
    parser.add_argument("--output-dir", type=Path, help="Defaults to outputs/<checkpoint dataset suffix>")
    parser.add_argument("--allow-subset", action="store_true", help="Allow a test split smaller than the reference benchmark")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 also works on Windows")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.checkpoint and args.model == "both":
        parser.error("--checkpoint requires --model base or small")
    if args.batch_size < 1 or args.threads < 1 or args.num_workers < 0:
        parser.error("batch-size/threads must be positive; num-workers must be non-negative")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if any(not group or Path(group).name != group or group in (".", "..")
           for group in (*args.image_dirs, args.label_dir)):
        parser.error("--image-dirs must name two distinct image folders alongside label/")
    if len(set((*args.image_dirs, args.label_dir))) != 3:
        parser.error("Image and label folders must be distinct")
    if args.normalization and args.compute_normalization:
        parser.error("--normalization and --compute-normalization are mutually exclusive")
    folder, suffix, _, _ = DATASETS[args.dataset]
    args.data_root = args.data_root or PROJECT_ROOT / "data" / folder
    args.output_dir = args.output_dir or PROJECT_ROOT / "outputs" / suffix
    return args


def main(argv=None):
    args = parse_args(argv)
    _, suffix, expected_tiles, statistics_tile_size = DATASETS[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install a CUDA PyTorch build or use --device cpu")
    torch.set_num_threads(args.threads)
    torch.manual_seed(36)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(PROJECT_ROOT))
    from cd.SChanger import schanger_base, schanger_small

    builders = {"base": schanger_base, "small": schanger_small}
    sizes = ("base", "small") if args.model == "both" else (args.model,)
    checkpoints = {size: args.checkpoint or args.weights_dir / f"schanger_{size}_{suffix}.pth" for size in sizes}
    for path in checkpoints.values():
        ensure_checkpoint(path, None if args.checkpoint else f"{WEIGHTS_URL}/{path.name}")
    normalization_path = args.normalization
    if not normalization_path and not args.compute_normalization:
        normalization_path = PROJECT_ROOT / "evaluation" / "normalization" / f"{suffix}.json"
    stats = (validate_normalization(json.loads(normalization_path.read_text(encoding="utf-8")))
             if normalization_path else train_statistics(args.data_root, tuple(args.image_dirs), statistics_tile_size))
    dataset = PairedTiles(args.data_root / args.split, stats, tuple(args.image_dirs), args.label_dir)
    if args.split == "test" and not args.allow_subset and len(dataset) != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} test tiles for {args.dataset}, found {len(dataset)}. "
                         "Check the split, or use --allow-subset for an intentional subset.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    print(f"Device: {device}; input pairs: {len(dataset.names)}; tiles: {len(dataset)}; pixels: {dataset.total_pixels}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "normalization.json", stats)
    report = {
        "status": "running", "dataset": args.dataset, "data_root": str(args.data_root.resolve()),
        "protocol": {"split": args.split, "image_dirs": list(args.image_dirs), "input_pairs": len(dataset.names),
                     "label_dir": args.label_dir, "expected_test_tiles": expected_tiles,
                     "full_test_count": args.split == "test" and len(dataset) == expected_tiles,
                     "tile_pairs": len(dataset), "pixels": dataset.total_pixels, "tile_size": TILE_SIZE,
                     "stride": TILE_SIZE, "resize": False, "tta": False, "threshold": args.threshold,
                     "threshold_comparison": ">", "tile_order": "filename, x, y",
                     "label_binarization": "label != 0",
                     "precision": "float32; TF32 disabled", "batch_size": args.batch_size,
                     "num_workers": args.num_workers, "threads": args.threads, "seed": 36,
                     "metric": "global pixel confusion matrix; positive-class binary F1; zero division = 0"},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "timm": version("timm"), "pillow": version("Pillow"),
                        "torchvision": version("torchvision"),
                        "device": str(device)},
        "normalization": stats, "normalization_file": str(normalization_path.resolve()) if normalization_path else None,
        "results": [],
    }
    write_json(args.output_dir / "results.json", report)
    try:
        for size, checkpoint in checkpoints.items():
            model = builders[size](args.dataset, pretrained=False)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or "state_dict" not in state:
                raise ValueError(f"Expected an official checkpoint containing 'state_dict': {checkpoint}")
            model.load_state_dict(state["state_dict"], strict=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            print(f"Evaluating SChanger-{size}: {checkpoint}", flush=True)
            counts, processed = evaluate(model, loader, device, args.threshold)
            if processed != len(dataset) or sum(counts) != dataset.total_pixels:
                raise RuntimeError("Incomplete evaluation: tile/pixel coverage check failed")
            result = {"model": size, "checkpoint": str(checkpoint.resolve()),
                      **metrics_from_confusion(counts), "seconds": time.perf_counter() - start}
            report["results"].append(result)
            write_json(args.output_dir / "results.json", report)
            print(json.dumps(result, indent=2), flush=True)
            del model, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        report["error"] = str(exc)
        write_json(args.output_dir / "results.json", report)
        raise
    report["status"] = "complete"
    write_json(args.output_dir / "results.json", report)
    print(f"Results: {args.output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
