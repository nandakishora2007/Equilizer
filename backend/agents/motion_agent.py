"""
Motion & Biological Consistency Agent.
Evaluates frame-to-frame temporal jitter, eye blinking rates, and unnatural head movements.
"""
'''
from pathlib import Path
from typing import Dict, Any, List

from backend.agents.base_agent import BaseAgent
from backend.core.constants import MOTION_AGENT
from backend.core.logger import logger


class MotionAgent(BaseAgent):
    def __init__(self, model_path: Path = None):
        super().__init__(name=MOTION_AGENT)
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Loads temporal motion/blink classification weights."""
        if self.model_path and self.model_path.exists():
            logger.info(f"[{self.name}] Loading motion/blink weights from {self.model_path}")
        else:
            logger.warning(
                f"[{self.name}] Model path not provided or found. Running in heuristic/fallback mode."
            )

    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates frame-to-frame temporal motion metrics and eye-aspect ratio (EAR).
        """
        frame_paths: List[str] = processed_data.get("frame_paths", [])

        if len(frame_paths) < 5:
            return self.format_result(
                score=0.0,
                confidence=0.2,
                details=["Insufficient frame sequence length for motion and blink analysis."],
            )

        logger.info(f"[{self.name}] Analyzing motion continuity across {len(frame_paths)} frames...")

        # --- Biological Consistency Check ---
        details = []
        # Fallback heuristic score: long frame sequences without blinks increase anomaly score
        estimated_blinks = 2
        
        if estimated_blinks == 0 and len(frame_paths) > 60:
            fake_score = 0.72
            confidence = 0.75
            details.append("Unnatural eye-blink rate detected: Zero eye blinks across extended video duration.")
        else:
            fake_score = 0.20
            confidence = 0.82
            details.append("Natural eye-blink patterns and smooth head motion transitions observed.")

        return self.format_result(
            score=fake_score,
            confidence=confidence,
            details=details,
        )


if __name__ == "__main__":
    agent = MotionAgent()
    print("MotionAgent Output:", agent.analyze({"frame_paths": ["f1.jpg", "f2.jpg"]}))
    '''


## New Version ##

"""
Motion & Temporal Continuity Analysis Agent.
"""

import cv2
import numpy as np
from typing import Dict, Any, List

from backend.agents.base_agent import BaseAgent
from backend.core.constants import MOTION_AGENT


class MotionAgent(BaseAgent):
    def __init__(self, model_path: str = None):
        super().__init__(name=MOTION_AGENT)

    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        frame_paths: List[str] = processed_data.get("frame_paths", [])

        if len(frame_paths) < 2:
            return self.format_result(
                score=0.0,
                confidence=0.1,
                details=["Insufficient frame sequence length for motion analysis."],
            )

        diffs = []
        prev_gray = None

        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))

            if prev_gray is not None:
                # Compute frame-to-frame mean absolute difference
                diff = cv2.absdiff(prev_gray, gray)
                diffs.append(np.mean(diff))

            prev_gray = gray

        if not diffs:
            return self.format_result(
                score=0.0,
                confidence=0.1,
                details=["Could not process sequence motion."],
            )

        motion_std = float(np.std(diffs))
        motion_mean = float(np.mean(diffs))

        # Deepfakes often display unnatural frame jitter or temporal freeze
        is_jittery = motion_std > 25.0 or (motion_mean < 1.0 and len(diffs) > 5)
        score = 0.78 if is_jittery else 0.12

        details = [
            f"Processed sequence of {len(diffs) + 1} frames for motion continuity.",
            f"Temporal Motion Fluctuation Index: {round(motion_std, 2)}",
            "Temporal flickering or motion discontinuity detected." if is_jittery else "Smooth frame-to-frame transition patterns."
        ]

        return self.format_result(
            score=score,
            confidence=0.75,
            details=details,
        )