# Notebook Provenance and Portability

The nine notebooks under `notebooks/` are retained as archival research records. They contain the original data-preparation, integrity-checking, training, and inference cells used during development.

## Machine-specific paths

A read-only audit found 118 occurrences of the original Windows project root in notebook source cells:

| Notebook | Source-cell occurrences |
|---|---:|
| `acdc_entire_training.ipynb` | 3 |
| `acdc_fog_training.ipynb` | 16 |
| `acdc_night_training.ipynb` | 4 |
| `acdc_rain_training.ipynb` | 7 |
| `dawn_entire_training.ipynb` | 2 |
| `dawn_fog_training.ipynb` | 13 |
| `dawn_rain_training.ipynb` | 10 |
| `yolov8_inference_pipeline.ipynb` | 30 |
| `combined_yolov8_training.ipynb` | 33 |

These paths are provenance from the original workstation. They are not credentials or secrets, but they are not portable.

The notebooks have not been subjected to a blind text replacement because path values occur in different contexts: dataset conversion, training configuration, model loading, backup staging, inference inputs, and output destinations. A mass replacement could create executable-looking cells with incorrect semantics while leaving historical cell outputs unchanged.

## Supported portable route

For a new training run, use the active relative dataset YAMLs, the documented Ultralytics `datasets_dir`, and the portable runner:

```cmd
python src\training\train_experiment.py --list
python src\training\train_experiment.py --experiment ACDC_FOG --dry-run
python src\training\train_experiment.py --experiment ACDC_FOG
```

Training presets are defined in `configs/TRAINING_PRESETS.json`. Each preset identifies its parameter source. Seven presets are derived from committed Ultralytics `args.yaml` files; ACDC Night and ACDC Rain use the explicit training calls retained in their notebooks because equivalent `args.yaml` files are not committed.

## Editing archival notebooks

If a notebook is adapted to another machine:

1. create a new working copy rather than overwriting the archival notebook;
2. configure dataset, model, and output roots explicitly;
3. rerun all dependent cells in order;
4. clear or regenerate stale outputs;
5. revalidate image-label matching and class IDs;
6. do not treat edited code plus old output as a reproduced experiment.

The `yolov8_inference_pipeline.ipynb` notebook additionally requires trained model weights, which are intentionally excluded from Git.
