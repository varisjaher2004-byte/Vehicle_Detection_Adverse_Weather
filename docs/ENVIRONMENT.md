# Verified Research Environment

This document records the software and hardware environment used for the verified dissertation experiments. It supports reproducibility but does not imply that identical results are guaranteed across different hardware, drivers, operating systems, random seeds, or library builds.

## Verified environment

Verification date: 20 August 2026

| Component | Verified value |
|---|---|
| Operating system | Microsoft Windows |
| Python | 3.12.10 |
| Ultralytics | 8.4.104 |
| PyTorch | 2.12.1+cu126 |
| Torchvision | 0.27.1+cu126 |
| CARLA Python API | 0.9.16 |
| NumPy | 2.5.1 |
| Pandas | 3.0.5 |
| OpenCV runtime | 5.0.0 (`opencv-python` package 5.0.0.93) |
| Matplotlib | 3.11.1 |
| PyYAML | 6.0.3 |
| Pillow | 12.3.0 |
| CUDA runtime reported by PyTorch | 12.6 |
| CUDA available | Yes |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |

## Installation

Create and activate a virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file includes the PyTorch CUDA 12.6 package index and the verified direct dependencies.

## CARLA requirement

The `carla` package installed through pip provides the Python client API. It does not install the complete CARLA simulator application.

The simulation scripts were developed against CARLA 0.9.16. A compatible CARLA 0.9.16 simulator installation must be started separately before running the scripts under `src/carla`.

The approved simulations use the `Town10HD_Opt` map. The principal capture configuration is 640×360 RGB at 10 sensor frames per second, with the synchronous world running at 20 ticks per second.

## Notebook interface

The verified virtual environment does not contain `jupyterlab` or `ipykernel` as explicitly installed packages. The notebooks may be opened through a compatible notebook interface, such as the Visual Studio Code Jupyter extension. Notebook-interface packages are not included in the verified runtime dependency claim.

## Reproducibility boundaries

The repository intentionally excludes:

- raw ACDC and DAWN datasets;
- trained model weights;
- standalone generated videos outside the embedded defence asset;
- large archives;
- local caches and temporary experiment runs.

These exclusions reduce repository size and respect third-party data-distribution restrictions. Users must obtain datasets through their original providers and reconstruct the documented YOLO directory structure.

Exact numerical reproduction may also depend on the original dataset split, annotation conversion, selected checkpoint, random state, GPU implementation, and library behaviour. The committed results are retained as research evidence from the completed experiments.
