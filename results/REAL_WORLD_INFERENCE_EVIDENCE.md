# Real-world inference evidence

This section records the saved real-world inference outputs produced by the
archived `notebooks/yolov8_inference_pipeline.ipynb` workflow. The notebook ran
Ultralytics prediction with a confidence threshold of 0.4 and saved annotated
JPEGs plus row-level detection logs.

## What is published

| Model scope | Evaluation image content | Annotated images | Location |
|---|---|---:|---|
| DAWN Entire | DAWN mixed-weather validation images | 120 | `DAWN/ENTIRE/qualitative_detections/` |
| Combined ACDC+DAWN | DAWN portion of the mixed validation union | 120 | `COMBINED/qualitative_detections/DAWN/` |

Seven `inference_results.csv` files are also stored with their corresponding
ACDC, DAWN or Combined experiment. Each row records the input filename, detected
class ID and confidence. The ACDC input filenames use `.png`, while Ultralytics
saved the annotated renderings as `.jpg`; this extension change does not indicate
a different input sample.

`REAL_WORLD_INFERENCE_INVENTORY.csv` is the integrity index for the complete
saved output collection. It contains 1,466 source artefact records: 1,459
annotated predictions and seven detection logs. Every record includes its logical
source path, model scope, dataset content, condition, source and repository byte
sizes and SHA-256 digests, duplicate-content count and publication decision.
Separate source and repository digests preserve the original CSV provenance while
also verifying Git's required LF line-ending normalisation. Machine-specific
absolute paths are deliberately omitted.

## Evidence boundary

The annotated images demonstrate that saved checkpoints were executed on
real-world adverse-weather inputs and produced visible detections. They are
qualitative inference evidence, not quantitative proof of accuracy. Use
`CORRECTED_2026-08-27/` for the dissertation's final numerical claims and the
historical training `results.csv` files only under the evidence rules in this
directory's main guide.

## Dataset licensing and attribution

DAWN is published by Mourad Kenk as **CC BY-NC 3.0**, DOI
[`10.17632/766ygrbt8y.3`](https://doi.org/10.17632/766ygrbt8y.3). The 240 DAWN
images in this repository are modified research outputs: this project added model
prediction boxes and labels. They are provided for non-commercial academic
inspection under the source dataset's attribution and non-commercial conditions.

The ACDC research-use licence prohibits sharing or distributing the dataset or
modified versions that directly contain its image data. For that reason, the
1,219 saved ACDC-background prediction files are documented by filename, size and
SHA-256 in the inventory but are not copied into the public repository. Obtain
ACDC directly from the [official dataset site](https://acdc.vision.ee.ethz.ch/)
and accept its [licence terms](https://acdc.vision.ee.ethz.ch/license) before
working with those images.

## Historical folder-name correction

Two archived local output folders were named `dawn_fog_test` and
`dawn_rain_test`. The notebook cells show that both the model and input paths were
ACDC Fog and ACDC Rain, respectively. Their 200 JPEGs are byte-identical to the
corresponding ACDC output folders. The associated CSV logs are therefore filed
under `ACDC/FOG/` and `ACDC/RAIN/`, not DAWN. The original folder labels remain in
the inventory so the provenance correction is transparent.

## Verification

Run the repository verifier from the project root:

```bash
python src/evaluation/verify_repository.py
```

It checks the published file count, allowed dataset content, destinations, byte
sizes and SHA-256 digests, as well as the quantitative evidence chain.
