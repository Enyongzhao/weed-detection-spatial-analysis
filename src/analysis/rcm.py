from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import cv2
from typing import Optional

def moving_average(values: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply a simple moving average for smoothing.

    Args:
        values: Input array.
        window_size: Smoothing window size.

    Returns:
        Smoothed array.
    """
    pad = window_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


def extract_frame(video_path: Path, frame_number: int = 200, save_path: Optional[Path] = None) -> Path:
    """
    Extract a specific frame from a video.

    Args:
        video_path: Path to the input video.
        frame_number: Frame index to extract.
        save_path: Path to save the extracted frame.

    Returns:
        Path to the saved frame image.
    """
    if save_path is None:
        save_path = Path("frame_sample.jpg")

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


def fit_crop_rows_ransac(crops: pd.DataFrame, n_rows: int = 3) -> list:
    """
    Fit crop rows using KMeans + RANSAC with the line model x = a*y + b.

    Args:
        crops: DataFrame containing crop detections.
        n_rows: Number of crop rows to fit.

    Returns:
        List of fitted RANSAC models.
    """
    models = []

    if len(crops) < n_rows:
        return models

    kmeans = KMeans(n_clusters=n_rows, random_state=0).fit(crops[["cx"]])
    cluster_labels = kmeans.labels_

    for i in range(n_rows):
        row_points = crops[cluster_labels == i]
        if len(row_points) < 10:
            continue

        model = RANSACRegressor(random_state=0)
        model.fit(row_points[["cy"]], row_points["cx"])
        models.append(model)

    return models


def get_line_params(model) -> tuple[float, float]:
    """
    Extract line parameters from a fitted RANSAC model.

    Returns:
        (a, b) for the line x = a*y + b
    """
    a = float(model.estimator_.coef_[0])
    b = float(model.estimator_.intercept_)
    return a, b


def project_to_line(point: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Project a point onto the line x = a*y + b.

    Args:
        point: Input point [x, y].
        a: Line slope parameter.
        b: Line intercept parameter.

    Returns:
        Projected point [x_proj, y_proj].
    """
    x0, y0 = float(point[0]), float(point[1])
    denom = 1.0 + a * a
    x_proj = (a * a * x0 + a * y0 + b) / denom
    y_proj = (a * x0 + y0 - a * b) / denom
    return np.array([x_proj, y_proj], dtype=float)


def line_distance(x0: float, y0: float, a: float, b: float) -> float:
    """
    Compute the perpendicular distance from a point to the line x = a*y + b.
    """
    return abs(x0 - a * y0 - b) / np.sqrt(1.0 + a * a)


def draw_crop_rows_on_frame(
    frame_path: Path,
    crops: pd.DataFrame,
    models: list,
    angle_name: str,
    output_path: Path,
) -> None:
    """
    Draw fitted crop rows on a sample frame.
    """
    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.scatter(crops["cx"], crops["cy"], s=8, c="yellow", alpha=0.6, label="Crop detections")

    if models:
        y_values = np.linspace(crops["cy"].min(), crops["cy"].max(), 500)
        for i, model in enumerate(models):
            a, b = get_line_params(model)
            x_values = a * y_values + b
            plt.plot(x_values, y_values, "r--", linewidth=2, label=f"Crop row {i + 1}")

    plt.title(f"Crop Row Detection - {angle_name}")
    plt.axis("off")
    plt.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Crop row visualization saved to: {output_path}")


def visualize_interrow_points(
    df: pd.DataFrame,
    models: list,
    angle_name: str,
    frame_path: Path,
    output_path: Path,
) -> None:
    """
    Visualize weed positions in the first inter-row region using relative position t.
    """
    weeds = df[df["cls_id"] == 1].drop_duplicates("track_id")
    crops = df[df["cls_id"] == 0]

    if len(models) < 2:
        print(f"[WARNING] Fewer than 2 crop rows detected for {angle_name}. Skipping inter-row point visualization.")
        return

    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    line_params = [get_line_params(m) for m in models]
    y_ref = float(crops["cy"].median())
    line_params.sort(key=lambda params: params[0] * y_ref + params[1])

    a1, b1 = line_params[0]
    a2, b2 = line_params[1]

    xs, ys, t_values = [], [], []
    for _, weed in weeds.iterrows():
        s1 = weed.cx - a1 * weed.cy - b1
        s2 = weed.cx - a2 * weed.cy - b2

        if s1 * s2 < 0:
            d1 = line_distance(weed.cx, weed.cy, a1, b1)
            d2 = line_distance(weed.cx, weed.cy, a2, b2)
            t = 100.0 * (d2 - d1) / (d1 + d2 + 1e-9)
            xs.append(weed.cx)
            ys.append(weed.cy)
            t_values.append(t)

    plt.figure(figsize=(12, 8))
    plt.imshow(frame)
    plt.scatter(crops["cx"], crops["cy"], s=5, c="yellow", alpha=0.35, label="Crop detections")
    scatter = plt.scatter(xs, ys, c=t_values, cmap="RdYlBu", s=12, alpha=0.9)
    plt.colorbar(scatter, label="Relative position t (-100 = left row, +100 = right row)")
    plt.title(f"Weed Relative Position Visualization - {angle_name}")
    plt.axis("off")
    plt.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Inter-row point visualization saved to: {output_path}")


def compute_interrow_distribution(
    df: pd.DataFrame,
    models: list,
    angle_name: str,
    output_dir: Path,
    smooth_k: int = 7,
    bins_count: int = 41,
    only_between_rows: bool = True,
) -> None:
    """
    Compute and plot weed distribution between adjacent crop rows.
    """
    crops = df[df["cls_id"] == 0]
    weeds = df[df["cls_id"] == 1].drop_duplicates("track_id")

    line_params = [get_line_params(m) for m in models]
    if len(line_params) < 2:
        print(f"[WARNING] Fewer than 2 crop rows detected for {angle_name}. Skipping inter-row distribution.")
        return

    y_ref = float(crops["cy"].median())
    line_params.sort(key=lambda params: params[0] * y_ref + params[1])

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(line_params) - 1):
        a1, b1 = line_params[i]
        a2, b2 = line_params[i + 1]

        t_values = []
        for _, weed in weeds.iterrows():
            if only_between_rows:
                s1 = weed.cx - a1 * weed.cy - b1
                s2 = weed.cx - a2 * weed.cy - b2
                if s1 * s2 >= 0:
                    continue

            d1 = line_distance(weed.cx, weed.cy, a1, b1)
            d2 = line_distance(weed.cx, weed.cy, a2, b2)
            t = 100.0 * (d2 - d1) / (d1 + d2 + 1e-9)
            t_values.append(t)

        if not t_values:
            print(f"[WARNING] No valid weeds found in inter-row {i + 1} for {angle_name}")
            continue

        bins = np.linspace(-100, 100, bins_count)
        hist, edges = np.histogram(t_values, bins=bins)
        x = (edges[:-1] + edges[1:]) / 2
        hist_smooth = moving_average(hist, window_size=smooth_k)

        plt.figure(figsize=(12, 6))
        plt.plot(x, hist, label="Weed count", alpha=0.55)
        plt.plot(x, hist_smooth, label=f"Smoothed (k={smooth_k})", linewidth=2)
        plt.axvline(0, linestyle="--", label="Midline (0)")
        plt.title(f"Weed Distribution - {angle_name} - Inter-row {i + 1}")
        plt.xlabel("Relative position (-100 = left crop row, +100 = right crop row)")
        plt.ylabel("Weed count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        output_path = output_dir / f"weed_distribution_{angle_name}_interrow{i + 1}.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[DONE] Inter-row distribution saved to: {output_path}")

