import os
from pathlib import Path


# PROJECT ROOT


PROJECT_ROOT = Path(os.environ.get("DMSC_PROJECT_ROOT", ".")).resolve()


# MODEL PATHS


MODELS = {

    "DAWN": PROJECT_ROOT / "models" / "dawn_entire_best.pt",

    "ACDC": PROJECT_ROOT / "models" / "acdc_entire_best.pt",

    "COMBINED": PROJECT_ROOT / "models" / "acdc+dawn_combined_best.pt",

}


# INPUT IMAGES

INPUT_FOLDER = PROJECT_ROOT / "outputs"

WEATHER_FOLDERS = [

    "clear",

    "light_rain",

    "heavy_rain",

    "light_fog",

    "dense_fog"

]


# OUTPUT


RESULTS_FOLDER = PROJECT_ROOT / "results"

CSV_FOLDER = RESULTS_FOLDER / "csv"

GRAPH_FOLDER = RESULTS_FOLDER / "graphs"

COMPARISON_FOLDER = RESULTS_FOLDER / "comparison"


# YOLO SETTINGS


IMAGE_SIZE = 640

CONFIDENCE = 0.25

DEVICE = "cpu"

SAVE_TXT = True

SAVE_CONF = True

SAVE_IMAGE = True


# CSV COLUMNS


CSV_COLUMNS = [

    "Model",

    "Weather",

    "Image",

    "Detections",

    "Average_Confidence",

    "Inference_Time",

    "FPS"

]
