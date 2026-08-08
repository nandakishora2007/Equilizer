import cv2
import numpy as np

from models.detector import get_detector
from utils.video_utils import sample_frames


class FaceAgent:
    """
    Specialized visual analysis agent.

    Pipeline:
        video
        -> sampled frames
        -> face detection
        -> face crops
        -> EfficientNet detector
        -> aggregated result
    """

    def __init__(self, sample_rate=1.0):
        self.sample_rate = sample_rate
        self.detector = get_detector()

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_detector.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar Cascade."
            )

    def analyze(self, video_path):
        frames = sample_frames(
            video_path,
            sample_rate=self.sample_rate
        )

        fake_probabilities = []

        frames_analyzed = 0
        faces_detected = 0

        for frame in frames:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) == 0:
                continue

            frame_h, frame_w = frame.shape[:2]

            valid_face_in_frame = False

            for x, y, w, h in faces:

                # Clamp coordinates to the actual image.
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_w, x + w)
                y2 = min(frame_h, y + h)

                if x2 <= x1 or y2 <= y1:
                    continue

                face = frame[
                    y1:y2,
                    x1:x2
                ]

                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                result = self.detector.predict(
                    face_rgb
                )

                fake_probabilities.append(
                    result["fake_probability"]
                )

                faces_detected += 1
                valid_face_in_frame = True

            if valid_face_in_frame:
                frames_analyzed += 1

        if not fake_probabilities:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "frames_analyzed": 0,
                "faces_detected": 0,
                "fake_probability": None,
                "reason": "No detectable faces were found."
            }

        fake_probability = float(
            np.mean(fake_probabilities)
        )

        fake_probability = float(
            np.clip(fake_probability, 0.0, 1.0)
        )

        confidence = max(
            fake_probability,
            1.0 - fake_probability
        )

        prediction = (
            "Fake"
            if fake_probability >= 0.5
            else "Real"
        )

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "frames_analyzed": frames_analyzed,
            "faces_detected": faces_detected,
            "fake_probability": fake_probability
        }