from ultralytics import YOLO
import time


class YOLOInference:

    def __init__(self, model_path):
        self.model = YOLO(str(model_path))

    def predict(self, image_path, output_folder):

        start_time = time.time()

        results = self.model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.25,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_folder),
            name="predictions",
            exist_ok=True,
            verbose=True
        )

        end_time = time.time()

        inference_time = end_time - start_time

        print("\n========================")
        print("Image :", image_path)
        print("Type  :", type(results))
        print("Length:", len(results))
        print("========================\n")

        if len(results) == 0:
            return None, inference_time

        return results[0], inference_time