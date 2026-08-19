def calculate_metrics(results, inference_time):

    if results is None:
        return {
            "Detections": 0,
            "Average_Confidence": 0,
            "Inference_Time": inference_time,
            "FPS": 1 / inference_time if inference_time > 0 else 0
        }

    detections = len(results.boxes)

    if detections > 0:
        avg_conf = float(results.boxes.conf.mean())
    else:
        avg_conf = 0

    fps = 1 / inference_time if inference_time > 0 else 0

    return {
        "Detections": detections,
        "Average_Confidence": avg_conf,
        "Inference_Time": inference_time,
        "FPS": fps
    }