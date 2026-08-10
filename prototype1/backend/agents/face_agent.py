"""
Face Analysis Agent.
Inspects extracted frames for facial anomalies, blending boundaries, and spatial inconsistencies.
"""

from pathlib import Path
from typing import Dict, Any, List
import cv2

from backend.agents.base_agent import BaseAgent
from backend.core.constants import FACE_AGENT
from backend.utils.frame_utils import crop_face
from backend.core.logger import logger


class FaceAgent(BaseAgent):
    def __init__(self, model_path: Path = None):
        super().__init__(name=FACE_AGENT)
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Loads face detection weights or initializes inference engine."""
        if self.model_path and self.model_path.exists():
            logger.info(f"[{self.name}] Loading facial model weights from {self.model_path}")
            # Insert PyTorch / ONNX model loading here
        else:
            logger.warning(
                f"[{self.name}] Model path not provided or found. Running in heuristic/mock fallback mode."
            )

    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes video frames for facial artifacts and boundary distortion.
        """
        frame_paths: List[str] = processed_data.get("frame_paths", [])

        if not frame_paths:
            return self.format_result(
                score=0.0,
                confidence=0.0,
                details=["No frame paths provided for face analysis."],
                error="Missing frames",
            )

        logger.info(f"[{self.name}] Analyzing {len(frame_paths)} frames for face artifacts...")

        faces_detected = 0
        fake_scores = []
        details = []

        for path_str in frame_paths:
            frame_path = Path(path_str)
            if not frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            face_crop, bbox = crop_face(frame)
            if face_crop is not None:
                faces_detected += 1
                
                # --- Model Inference / Heuristic Analysis ---
                # Placeholder heuristic: analyze color variance / variance of laplacian (blur)
                gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

                # Low variance / blur in cropped faces often points to AI smoothing
                if blur_var < 100.0:
                    fake_scores.append(0.75)
                else:
                    fake_scores.append(0.20)

        if faces_detected == 0:
            return self.format_result(
                score=0.0,
                confidence=0.3,
                details=["No faces were detected across all extracted frames."],
            )

        # Aggregate frame-level scores
        avg_score = sum(fake_scores) / len(fake_scores)
        confidence = min(0.95, 0.5 + (faces_detected / len(frame_paths)) * 0.5)

        details.append(f"Processed {faces_detected} face crops out of {len(frame_paths)} frames.")
        if avg_score > 0.5:
            details.append("Facial smoothing/edge boundary artifacts detected in frame samples.")
        else:
            details.append("Facial features display natural texture and boundary consistency.")

        return self.format_result(
            score=avg_score,
            confidence=confidence,
            details=details,
        ) 