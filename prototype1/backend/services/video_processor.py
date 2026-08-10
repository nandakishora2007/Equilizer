"""
Video processing service that extracts frame sequences and audio channels 
for multi-agent analysis.
"""

from pathlib import Path
from typing import Dict, Any, List

from backend.utils.frame_utils import extract_frames
from backend.utils.audio_utils import extract_audio_from_video
from backend.core.logger import logger


class VideoProcessor:
    def __init__(self, fps_sample_rate: float = 1.0, max_frames: int = 50):
        self.fps_sample_rate = fps_sample_rate
        self.max_frames = max_frames

    def process(self, media_path: Path, job_dir: Path) -> Dict[str, Any]:
        """
        Processes media file and populates the job directory with artifacts.

        Returns a payload dictionary containing paths to frames and extracted audio.
        """
        logger.info(f"Starting media processing for: {media_path.name}")

        frames_dir = job_dir / "frames"
        audio_path = job_dir / "extracted_audio.wav"

        # 1. Extract frames
        frame_paths: List[Path] = extract_frames(
            video_path=media_path,
            output_dir=frames_dir,
            fps_sample_rate=self.fps_sample_rate,
            max_frames=self.max_frames,
        )

        # 2. Extract audio channel
        extracted_audio = extract_audio_from_video(
            video_path=media_path,
            output_audio_path=audio_path,
        )

        processed_data = {
            "media_path": str(media_path),
            "job_dir": str(job_dir),
            "frame_paths": [str(p) for p in frame_paths],
            "frame_count": len(frame_paths),
            "audio_path": str(extracted_audio) if extracted_audio else None,
            "has_audio": extracted_audio is not None and extracted_audio.exists(),
        }

        logger.info(
            f"Video processing finished | Frames: {len(frame_paths)} | "
            f"Has Audio: {processed_data['has_audio']}"
        )
        return processed_data
    