from ultralytics import YOLO

def export_model_any_size(model_path: str):
    model = YOLO(model_path)
    model.export(
        format="engine",
        dynamic=True,
        imgsz=[1280, 1280],  # min, optimal, max image sizes
        #half=True,
        #task='pose'
        #int8=True
    )

# Example usage
export_model_any_size("football-ball-detection.pt")
