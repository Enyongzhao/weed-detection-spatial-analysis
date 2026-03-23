from pathlib import Path
import csv
import cv2
from ultralytics import YOLO


def run_inference(
    video_path: Path,
    model_path: Path,
    output_csv: Path,
    output_video: Path,
    fps_scale: float = 0.3,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    resize_scale: float = 0.5,
) -> None:
    """
    Run object detection and tracking on a video using a trained YOLO model.

    Args:
        video_path: Path to the input video.
        model_path: Path to the trained YOLO model weights.
        output_csv: Path to save detection results in CSV format.
        output_video: Path to save the annotated output video.
        fps_scale: Scale factor for output video FPS.
        conf_threshold: Confidence threshold for detection.
        iou_threshold: IoU threshold for tracking.
        resize_scale: Scale factor for output video resolution.
    """
    if not video_path.exists():
        print(f"[ERROR] Input video not found: {video_path}")
        return

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        print(f"[ERROR] Failed to read video properties: {video_path}")
        return

    new_width = int(width * resize_scale)
    new_height = int(height * resize_scale)
    new_fps = max(1, int(fps * fps_scale))
    frame_interval = max(1, int(1 / fps_scale))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        new_fps,
        (new_width, new_height),
    )

    frame_id = 0
    saved_frame_id = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_id", "track_id", "cls_id", "cx", "cy"])

        for result in model.track(
            source=str(video_path),
            conf=conf_threshold,
            iou=iou_threshold,
            tracker="bytetrack.yaml",
            stream=True,
            show=False,
        ):
            annotated_frame = result.plot()

            if frame_id % frame_interval == 0:
                resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
                video_writer.write(resized_frame)

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()

                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy()
                    else:
                        track_ids = range(len(boxes))

                    for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        writer.writerow(
                            [
                                saved_frame_id,
                                int(track_id),
                                int(cls_id),
                                float(cx),
                                float(cy),
                            ]
                        )

                saved_frame_id += 1

            frame_id += 1

    video_writer.release()

    print(f"[DONE] Detection results saved to: {output_csv}")
    print(f"[DONE] Annotated video saved to: {output_video}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    video_path = project_root / "data" / "videos" / "camera_90" / "GX010844.MP4"
    model_path = project_root / "models" / "best.pt"
    output_csv = project_root / "results" / "csv" / "detection_results_camera90.csv"
    output_video = project_root / "results" / "videos" / "detection_results_camera90.avi"

    run_inference(
        video_path=video_path,
        model_path=model_path,
        output_csv=output_csv,
        output_video=output_video,
        fps_scale=0.3,
    )



