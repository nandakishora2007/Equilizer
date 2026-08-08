from utils.audio_utils import (
    audio_features,
    cleanup_audio,
    load_audio
)


class AudioAgent:
    """
    Lightweight supporting audio agent.

    This is NOT a trained audio deepfake classifier.
    Its output is weak supporting evidence for FusionAgent.
    """

    def analyze(self, video_path):

        audio_path = None

        try:
            audio, sample_rate, audio_path = load_audio(
                video_path
            )

            if audio is None:
                return {
                    "prediction": "Unknown",
                    "confidence": 0.0,
                    "fake_probability": None,
                    "reason": "No audio track detected."
                }

            features = audio_features(
                audio,
                sample_rate
            )

            if features is None:
                return {
                    "prediction": "Unknown",
                    "confidence": 0.0,
                    "fake_probability": None,
                    "reason": "Audio could not be analyzed."
                }

            fake_probability = self._heuristic_score(
                features
            )

            if fake_probability >= 0.5:
                prediction = "Fake"
                confidence = fake_probability
            else:
                prediction = "Real"
                confidence = 1.0 - fake_probability

            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(fake_probability),
                "features": features,
                "reason": (
                    "MFCC and spectral heuristics were used as "
                    "supporting evidence only; this is not a "
                    "trained audio deepfake probability."
                )
            }

        finally:
            cleanup_audio(audio_path)

    @staticmethod
    def _heuristic_score(features):
        """
        Return a weak normalized supporting fake score in [0, 0.5].

        This deliberately prevents the heuristic audio agent from
        overpowering the trained visual detector.
        """

        score = 0.0

        if features["spectral_flatness_mean"] > 0.15:
            score += 0.20

        if features["zcr_mean"] > 0.15:
            score += 0.15

        if features["mfcc_std"] < 25:
            score += 0.15

        return min(score, 0.5)