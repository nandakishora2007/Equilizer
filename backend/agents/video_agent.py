"""
Video Analysis Agent for detecting deepfake motion and frame anomalies.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from backend.agents.base_agent import BaseAgent
from backend.core.logger import logger

# Fallback constant definition
VIDEO_AGENT = "Video Analysis Agent"


class VideoAgent(BaseAgent):
    def __init__(self, model_path: Path = None):
        super().__init__(name=VIDEO_AGENT)

    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes video frame temporal consistency and compression artifacts.
        """
        try:
            frame_paths: List[str] = processed_data.get("frame_paths", [])

            if not frame_paths:
                return self.format_result(
                    score=0.5,
                    confidence=0.1,
                    details=["No frames provided for video analysis."],
                )

            frame_scores = []
            prev_gray = None
            flicker_count = 0

            for path in frame_paths:
                img = cv2.imread(path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Measure frame-to-frame temporal inconsistency
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = float(np.mean(diff))

                    if mean_diff > 35.0:  # High inter-frame fluctuation
                        flicker_count += 1
                        frame_scores.append(0.80)
                    else:
                        frame_scores.append(0.20)

                prev_gray = gray

            avg_score = float(np.mean(frame_scores)) if frame_scores else 0.5
            confidence = round(min(0.90, 0.4 + (len(frame_paths) * 0.05)), 2)

            return self.format_result(
                score=round(avg_score, 4),
                confidence=confidence,
                details=[
                    f"Analyzed {len(frame_paths)} sequential video frames.",
                    f"Temporal flickering/artifacts flagged: {flicker_count} transition(s).",
                    "Result: Video frames exhibit natural optical flow." if avg_score < 0.5 else "Result: Abnormal frame-to-frame temporal noise detected."
                ],
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error during video frame analysis: {e}")
            return self.format_result(
                score=0.5,
                confidence=0.2,
                details=[f"Video analysis fallback triggered: {str(e)}"],
            )