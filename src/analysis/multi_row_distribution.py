from pathlib import Path
import pandas as pd

from rcm import (
    extract_frame,
    fit_crop_rows_ransac,
    draw_crop_rows_on_frame,
    compute_interrow_distribution,
    visualize_interrow_points,
)


def analyze_multi_row_view(
    csv_path: Path,
    video_path: Path,
    angle_name: str,
    frame_number: int = 200,
    n_rows: int = 3,
) -> None:
    """
    Analyze multi-row crop views such as 10°, 30°, and 50°.
    """
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "results" / "plots" / angle_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    crops = df[df["cls_id"] == 0]

    if crops.empty:
        print(f"[WARNING] No crop detections found for {angle_name}")
        return

    models = fit_crop_rows_ransac(crops, n_rows=n_rows)

    frame_path = extract_frame(
        video_path=video_path,
        frame_number=frame_number,
        save_path=output_dir / f"frame_{angle_name}.jpg",
    )

    draw_crop_rows_on_frame(
        frame_path=frame_path,
        crops=crops,
        models=models,
        angle_name=angle_name,
        output_path=output_dir / f"row_demo_{angle_name}.png",
    )

    compute_interrow_distribution(
        df=df,
        models=models,
        angle_name=angle_name,
        output_dir=output_dir,
        smooth_k=7,
        bins_count=41,
        only_between_rows=True,
    )

    visualize_interrow_points(
        df=df,
        models=models,
        angle_name=angle_name,
        frame_path=frame_path,
        output_path=output_dir / f"weed_color_visual_{angle_name}.png",
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    tasks = [
        {
            "angle_name": "10deg",
            "csv_path": project_root / "results" / "csv" / "detection_results_camera10.csv",
            "video_path": project_root / "data" / "videos" / "camera_10" / "GX011076.MP4",
        },
        {
            "angle_name": "30deg",
            "csv_path": project_root / "results" / "csv" / "detection_results_camera30.csv",
            "video_path": project_root / "data" / "videos" / "camera_30" / "GX010720.MP4",
        },
        {
            "angle_name": "50deg",
            "csv_path": project_root / "results" / "csv" / "detection_results_camera50.csv",
            "video_path": project_root / "data" / "videos" / "camera_50" / "GX011048.MP4",
        },
    ]

    for task in tasks:
        analyze_multi_row_view(
            csv_path=task["csv_path"],
            video_path=task["video_path"],
            angle_name=task["angle_name"],
            frame_number=200,
            n_rows=3,
        )