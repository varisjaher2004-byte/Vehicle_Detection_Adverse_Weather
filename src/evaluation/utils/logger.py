import logging
from pathlib import Path


def setup_logger(log_folder):

    log_folder = Path(log_folder)

    log_folder.mkdir(parents=True, exist_ok=True)

    log_file = log_folder / "evaluation.log"

    logger = logging.getLogger("YOLO_Evaluation")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger