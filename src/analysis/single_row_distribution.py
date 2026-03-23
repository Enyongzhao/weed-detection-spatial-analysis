from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from typing import Optional

def moving_average(values: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a simple moving average for smoothing.
    """
    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


def extract_frame(video_path: Path, frame_number: int = 100, save_path: Optional[Path] = None) -> Path:
    """
    Extract a specific frame from a video.
    """
    if save_path is None:
        save_path = Path("example_frame.jpg")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"[ERROR] Failed to read frame {frame_number} from {video_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), frame)
    print(f"[DONE] Extracted frame saved to: {save_path}")
    return save_path


def draw_row_demo(
    frame_path: Path,
    crops: pd.DataFrame,
    row_center: float,
    angle_name: str,
    output_path: Path,
) -> None:
    """
    Draw the estimated single crop row on a sample frame.
    """
    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.axvline(row_center, color="red", linestyle="-", linewidth=2, label="Crop row")
    plt.scatter(crops["cx"], crops["cy"], s=10, c="yellow", alpha=0.6, label="Crop detections")

    plt.title(f"Single Crop Row Visualization - {angle_name}")
    plt.axis("off")
    plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Single-row visualization saved to: {output_path}")


def plot_single_row_distribution(
    df: pd.DataFrame,
    angle_name: str,
    output_dir: Path,
    video_path: Optional[Path] = None,
    frame_number: int = 100,
    smooth_k: int = 7,
) -> None:
    """
    Plot weed distribution for near-single-row views such as 70° and 90°.
    """
    crops = df[df["cls_id"] == 0]
    weeds = df[df["cls_id"] == 1].drop_duplicates("track_id")

    if crops.empty or weeds.empty:
        print(f"[WARNING] No crop or weed detections found for {angle_name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    kmeans = KMeans(n_clusters=1, random_state=0).fit(crops[["cx"]])
    row_center = kmeans.cluster_centers_.flatten()[0]

    left_bound = df["cx"].min()
    right_bound = df["cx"].max()

    weeds_in_range = weeds[(weeds["cx"] >= left_bound) & (weeds["cx"] <= right_bound)].copy()
    if weeds_in_range.empty:
        print(f"[WARNING] No weeds found in valid range for {angle_name}")
        return

    weeds_in_range["cx_norm"] = (
        (weeds_in_range["cx"] - left_bound) / (right_bound - left_bound) * 200 - 100
    )

    bins = np.linspace(-100, 100, 41)
    hist, edges = np.histogram(weeds_in_range["cx_norm"], bins=bins)
    hist_smooth = moving_average(hist, window_size=smooth_k)
    x = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(12, 6))
    plt.plot(x, hist, label="Weed count", alpha=0.6)
    plt.plot(x, hist_smooth, label=f"Smoothed (k={smooth_k})", linewidth=2)

    plt.axvline(-100, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1, label="Center")
    plt.axvline(100, color="gray", linestyle="--", linewidth=1)

    crop_row_norm = (row_center - left_bound) / (right_bound - left_bound) * 200 - 100
    plt.axvline(crop_row_norm, color="red", linestyle="-", linewidth=2, label="Crop row")

    plt.title(f"Weed Distribution - {angle_name} (Single Row)")
    plt.xlabel("Relative position (Left = -100, Right = +100)")
    plt.ylabel("Weed count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    distribution_path = output_dir / f"weed_distribution_{angle_name}_singlerow.png"
    plt.savefig(distribution_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Single-row distribution saved to: {distribution_path}")

    if video_path is not None:
        frame_path = extract_frame(
            video_path=video_path,
            frame_number=frame_number,
            save_path=output_dir / f"frame_{angle_name}.jpg",
        )
        draw_row_demo(
            frame_path=frame_path,
            crops=crops,
            row_center=row_center,
            angle_name=angle_name,
            output_path=output_dir / f"row_demo_{angle_name}.png",
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    tasks = [
        {
            "angle_name": "70deg",
            "csv_path": project_root / "results" / "csv" / "detection_results_camera70.csv",
            "video_path": project_root / "data" / "videos" / "camera_70" / "GX011137.MP4",
        },
        {
            "angle_name": "90deg",
            "csv_path": project_root / "results" / "csv" / "detection_results_camera90.csv",
            "video_path": project_root / "data" / "videos" / "camera_90" / "GX010844.MP4",
        },
    ]

    for task in tasks:
        if not task["csv_path"].exists():
            print(f"[WARNING] CSV not found: {task['csv_path']}")
            continue

        df = pd.read_csv(task["csv_path"])
        plot_single_row_distribution(
            df=df,
            angle_name=task["angle_name"],
            output_dir=project_root / "results" / "plots" / task["angle_name"],
            video_path=task["video_path"],
            frame_number=200,
            smooth_k=7,
        )