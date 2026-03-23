from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
def extract_frames(video_path: Path, output_dir: Path, num_frames: int = 5) -> None:
    """
    Extract a fixed number of frames from a single video.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        num_frames: Number of frames to extract evenly across the video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"[WARNING] Could not read total frame count from: {video_path}")
        cap.release()
        return

    if total_frames < num_frames:
        print(
            f"[INFO] Video {video_path.name} has fewer than {num_frames} frames. "
            f"All available frames will be extracted."
        )
        frame_indices = range(total_frames)
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    saved_count = 0
    video_name = video_path.stem

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            frame_filename = f"{video_name}_frame{saved_count + 1}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        else:
            print(f"[WARNING] Failed to read frame {idx} from {video_path.name}")

    cap.release()
    print(f"[DONE] Extracted {saved_count} frames from {video_path.name}")


def batch_extract(input_dir: Path, output_dir: Path, num_frames: int = 5) -> None:
    """
    Extract frames from all MP4 videos in a directory.

    Args:
        input_dir: Directory containing input MP4 videos.
        output_dir: Directory to save extracted frames.
        num_frames: Number of frames to extract from each video.
    """
    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    video_files = sorted(
        {p.resolve() for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
    )

    if not video_files:
        print(f"[WARNING] No MP4 files found in: {input_dir}")
        return

    for video_path in video_files:
        extract_frames(video_path, output_dir, num_frames)

if __name__ == "__main__":
    # Example:
    # data/videos/camera_90/
    # data/frames/camera_90/

    input_dir = PROJECT_ROOT / "data/videos/camera_90"
    output_dir = PROJECT_ROOT / "data/frames/camera_90"

    num_frames = 5

    batch_extract(input_dir, output_dir, num_frames)