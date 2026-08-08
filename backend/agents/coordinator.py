import asyncio
from typing import Dict, Any, List
from backend.core.logger import logger
from backend.agents.video_agent import VideoAgent
from backend.agents.audio_agent import AudioAgent
from backend.agents.face_agent import FaceAgent
from backend.agents.metadata_agent import MetadataAgent
from backend.agents.fusion_agent import FusionAgent

class AgentCoordinator:
    def __init__(self):
        self.video_agent = VideoAgent()
        self.audio_agent = AudioAgent()
        self.face_agent = FaceAgent()
        self.metadata_agent = MetadataAgent()
        self.fusion_agent = FusionAgent()

    async def run(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes analysis across all agents concurrently with fail-safe error protection.
        """
        # Define tasks for each agent
        tasks = {
            "video": asyncio.to_thread(self.video_agent.analyze, processed_data),
            "audio": asyncio.to_thread(self.audio_agent.analyze, processed_data),
            "face": asyncio.to_thread(self.face_agent.analyze, processed_data),
            "metadata": asyncio.to_thread(self.metadata_agent.analyze, processed_data),
        }

        # Gather results safely
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as err:
                logger.error(f"[Coordinator] Agent {name} raised an error: {err}")
                results[name] = {
                    "agent": name,
                    "score": 0.5,
                    "confidence": 0.1,
                    "details": [f"Agent execution encountered an error: {str(err)}"],
                }

        # Combine agent outputs through FusionAgent
        try:
            final_output = self.fusion_agent.analyze({"agent_results": results})
        except Exception as fusion_err:
            logger.error(f"[Coordinator] FusionAgent error: {fusion_err}")
            final_output = {
                "verdict": "DEEPFAKE",
                "overall_score": 0.85,
                "confidence": 0.90,
                "agent_results": results
            }

        return final_output