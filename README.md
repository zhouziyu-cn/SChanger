## SChanger: Change Detection from a Semantic Change and Spatial Consistency Perspective (jstars 2025)

<h5 align="left">Ziyu Zhou, Keyan Hu, Yutian Fang and Xiaoping Rui</h5>

[[`Paper`](https://arxiv.org/abs/2503.20734)]



<p align="center">
    <img src="./assets/compare.png" width="1200">
</p>

### News

- 2025/03, This paper is accepted by jstars.

### Catalog

Paper F1 scores (%) from Tables III-VIII of the
[published article](https://doi.org/10.1109/JSTARS.2025.3555849).
The current WHU-CD small result is shown in parentheses.

| Model | LEVIR-CD | LEVIR-CD+ | S2Looking | CDD | SYSU-CD | WHU-CD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SChanger-small | 92.45 | 86.20 | 68.20 | 95.75 | 84.58 | 93.15 (93.51) |
| SChanger-base | 92.87 | 86.43 | 68.95 | 97.62 | 84.17 | 93.20 |



---------------------



### Evaluation

Evaluate the official checkpoints with 256x256 non-overlapping tiles:

```sh
python -m pip install -r requirements-eval.txt
python evaluation/evaluate.py --dataset levir-cd
python evaluation/evaluate.py --dataset levir-cd+
python evaluation/evaluate.py --dataset s2looking
python evaluation/evaluate.py --dataset cdd
python evaluation/evaluate.py --dataset sysu-cd
python evaluation/evaluate.py --dataset whu-cd
```

See [evaluation/README.md](evaluation/README.md) for dataset preparation, checkpoint
links, training normalization, CPU/CUDA options and the evaluation protocol.

### Citation

If you use SChanger models in your research, we hope you can kindly cite the following papers:
```text
@article{zhou2025schanger,
  title={SChanger: Change Detection from a Semantic Change and Spatial Consistency Perspective},
  author={Zhou, Ziyu and Hu, Keyan and Fang, Yutian and Rui, Xiaoping},
  journal={arXiv preprint arXiv:2503.20734},
  year={2025}
}
```

### License
This code is released under the [Apache License 2.0](https://github.com/Z-Zheng/ChangeStar/blob/master/LICENSE).

Copyright (c) Ziyu Zhou. All rights reserved.
