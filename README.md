# iPack Evaluator

[![CI](https://github.com/yblei/ipack_evaluator/actions/workflows/main.yml/badge.svg)](https://github.com/yblei/ipack_evaluator/actions/workflows/main.yml)

Packing benchmark, created as a part of the [iPack: Intuitive Bin Packing with Large Language Models](https://github.com/yblei/ipack). (Blei et al, 2026)

<div align="center">
<img src="assets/mujoco.png" width="250 em" style="padding:2em">
</div>

This benchmark selects a random subset of objects from the [Nvidia-HOPE](https://github.com/swtyree/hope-dataset) dataset as well as fragile objects form the HOPE-extension, recorded as a part of this project. The baselines receive a list of objects and a return a list of placements. A mujoco based **physics simulation** optionally check for packing stability. We also compute the alignment with **human packing preference** through the **C metric** (see publication for details). 


## Installation

Tested on Python 3.11.
Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Run the example evaluator:

```bash
python evaluate_example.py
```

That example defines a simple placement routine and passes it into the evaluator via `run_exp(...)`.

## Supported Baselines

We provide integrations to the following baselines in the `scripts/` directory:
- [Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning](https://github.com/Xiong5Heng/GOPT) (Xiong et al, 2024)
- [Optimizing Three-Dimensional Bin Packing Through Simulation](https://github.com/jerry800416/3D-bin-packing) (Dube et al, 2006)
- [iPack: Intuitive Bin Packing with Large Language Models](https://github.com/yblei/ipack) (Blei et al, 2026)



## Citation
When used in academic work, please cite:
```bash
Arxive citation to be added here
```

## License

Licensed under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).