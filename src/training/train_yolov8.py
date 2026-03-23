from pathlib import Path
from ultralytics import YOLO


def train_model() -> None:
    """
    Train a YOLOv8 model on the custom crop and weed dataset.
    """
    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "models" / "yolov8n.pt"
    data_yaml = project_root / "dataset" / "data.yaml"
    output_dir = project_root / "results"

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return

    if not data_yaml.exists():
        print(f"[ERROR] Dataset config file not found: {data_yaml}")
        return

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        project=str(output_dir),
        name="yolov8_weed_detection"
    )

    print("[DONE] Training completed.")


if __name__ == "__main__":
    train_model()