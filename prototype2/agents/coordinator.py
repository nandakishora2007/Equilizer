from agents.audio_agent import AudioAgent
from agents.face_agent import FaceAgent
from agents.fusion_agent import FusionAgent


class Coordinator:
    """
    Orchestrates the specialized agents.

    The Coordinator does not perform detection itself.
    """

    def __init__(self):
        self.face_agent = FaceAgent(
            sample_rate=1.0
        )

        self.audio_agent = AudioAgent()

        self.fusion_agent = FusionAgent()

    def analyze(self, video_path):
        """
        Run all agents and return the final system result.
        """

        face_result = self.face_agent.analyze(
            video_path
        )

        audio_result = self.audio_agent.analyze(
            video_path
        )

        fusion_result = self.fusion_agent.analyze(
            face_result,
            audio_result
        )

        return {
            "prediction": fusion_result["prediction"],
            "confidence": fusion_result["confidence"],
            "explanation": fusion_result["explanation"],
            "agents": {
                "Face Agent": face_result,
                "Audio Agent": audio_result
            }
        }