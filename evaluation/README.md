# SChanger evaluation

Evaluate the published base and small checkpoints on six change detection
datasets.

## Installation

Install the [PyTorch build](https://pytorch.org/get-started/locally/) for your
device, then run:

```sh
python -m pip install -r requirements-eval.txt
```

Tested with Python 3.11.16, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128,
timm 1.0.29, NumPy 2.4.6 and Pillow 12.3.0.

## Data and weights

| `--dataset` | Data folder | Weight suffix | Test tiles (256x256) |
| --- | --- | --- | ---: |
| `levir-cd` | `data/LEVIR-CD` | `levir` | 2048 |
| `levir-cd+` | `data/LEVIR-CD+` | `levir_plus` | 5568 |
| `s2looking` | `data/S2Looking` | `s2looking` | 16000 |
| `cdd` | `data/CDD` | `cdd` | 3000 |
| `sysu-cd` | `data/SYSU-CD` | `sysu` | 4000 |
| `whu-cd` | `data/WHU-CD` | `whu` | 2760 |

Download [the published weights](https://huggingface.co/Zy-Zhou/schanger/tree/main)
to `weights/schanger_{base,small}_{suffix}.pth`.

Dataset sources: [LEVIR-CD](https://justchenhao.github.io/LEVIR/),
[LEVIR-CD+ and S2Looking](https://github.com/S2Looking/Dataset),
[SYSU-CD](https://github.com/liumency/SYSU-CD),
[CDD and WHU-CD](https://github.com/ChengxiHAN/C2F-SemiCD-and-C2FNet#dataset-download).

For WHU-CD, we use the preprocessed 256x256 `WHU-CD256-HANet` split linked in
the [C2FNet repository](https://github.com/ChengxiHAN/C2F-SemiCD-and-C2FNet#dataset-download).
It contains **4536/504/2760** train/validation/test pairs and is distributed as
`3-WHU-CD256-Train,Val,Test-HANet.zip`
([download](https://pan.baidu.com/s/16g3H1UsDMgqmXaVjiE319Q?pwd=6969), password `6969`).
Use this prepared split; other WHU splits have different test images.

Only the test split is required with the included training normalization:

```text
data/<dataset>/test/
  A/       # first-date RGB images
  B/       # second-date RGB images
  label/   # single-channel masks
```

## Run

```sh
python evaluation/evaluate.py --dataset levir-cd
python evaluation/evaluate.py --dataset levir-cd+
python evaluation/evaluate.py --dataset s2looking
python evaluation/evaluate.py --dataset cdd
python evaluation/evaluate.py --dataset sysu-cd
python evaluation/evaluate.py --dataset whu-cd
```

Both model sizes run by default. Examples:

```sh
python evaluation/evaluate.py --dataset cdd --model base --device cuda:0
python evaluation/evaluate.py --dataset sysu-cd --model small --device cpu
python evaluation/evaluate.py --dataset sysu-cd --data-root /datasets/SYSU-CD --image-dirs time1 time2
python evaluation/evaluate.py --dataset cdd --data-root /datasets/CDD --image-dirs t1 t2 --label-dir label
python evaluation/evaluate.py --dataset whu-cd --model small --checkpoint weights/custom.pth
```

Default paths are relative to the repository. Explicit relative paths are
relative to the current directory. Use `--batch-size` and `--num-workers` to
configure loading, `--split val` for validation, and `--allow-subset` only for
an intentional subset. Full test tile counts are checked by default.

## Normalization

`evaluation/normalization/<suffix>.json` provides separate RGB statistics for
A and B: averages of per-training-image means and population standard deviations
in [0, 1]. Validation and test images do not contribute.

| Dataset | Training samples used for statistics |
| --- | --- |
| LEVIR-CD | 445 original 1024x1024 pairs |
| LEVIR-CD+ | 10192 cropped 256x256 pairs; author's saved statistics |
| S2Looking | 56000 cropped 256x256 pairs; author's saved statistics |
| CDD | 10000 pairs; author's saved statistics |
| SYSU-CD | 12000 pairs |
| WHU-CD | 4536 pairs from the WHU-CD256-HANet archive |

Use `--normalization path/to/file.json` for another saved configuration, or
`--compute-normalization` to calculate statistics from `train/`. LEVIR-CD+ and
S2Looking use 256x256 training crops for this calculation.

## Measured results

Full-test positive-class F1 percentages, measured on 2026-09-13 using the
published checkpoints and the protocol above:

| Dataset | Test tiles | Base F1 (%) | Small F1 (%) |
| --- | ---: | ---: | ---: |
| LEVIR-CD | 2048 | 92.8831 | 92.4391 |
| LEVIR-CD+ | 5568 | 86.4259 | 86.1960 |
| CDD | 3000 | 97.6225 | 95.7662 |
| SYSU-CD | 4000 | 84.1692 | 84.5759 |
| WHU-CD | 2760 | 93.2020 | 93.5141 |
| S2Looking | 16000 | 68.9598 | 68.1994 |
