# Archival Research Notebooks

These nine source-only notebooks preserve the development route used for the original experiments. Their filenames identify the dataset, condition and purpose directly.

| Notebook | Purpose |
|---|---|
| [`acdc_entire_training.ipynb`](acdc_entire_training.ipynb) | Prepared ACDC Entire training workflow |
| [`acdc_fog_training.ipynb`](acdc_fog_training.ipynb) | Prepared ACDC Fog training workflow |
| [`acdc_night_training.ipynb`](acdc_night_training.ipynb) | Prepared ACDC Night training workflow and parameter provenance |
| [`acdc_rain_training.ipynb`](acdc_rain_training.ipynb) | Prepared ACDC Rain training workflow and parameter provenance |
| [`dawn_entire_training.ipynb`](dawn_entire_training.ipynb) | Historical DAWN Entire workflow |
| [`dawn_fog_training.ipynb`](dawn_fog_training.ipynb) | Historical DAWN Fog workflow |
| [`dawn_rain_training.ipynb`](dawn_rain_training.ipynb) | Historical DAWN Rain workflow |
| [`combined_yolov8_training.ipynb`](combined_yolov8_training.ipynb) | Historical Combined ACDC+DAWN workflow |
| [`yolov8_inference_pipeline.ipynb`](yolov8_inference_pipeline.ipynb) | Multi-condition inference and output-processing workflow |

The public notebooks have cleared execution counts and outputs and retain machine-specific paths from the original workstation. They are provenance records, not the supported portable route. Use [`train_experiment.py`](../src/training/train_experiment.py) and the workflows in [Getting started](../docs/GETTING_STARTED.md) for a new run.

There is no separate ACDC Snow notebook. Its accepted parameters are preserved through `results/ACDC/SNOW/args.yaml`, the active YAML and the verified training preset.
