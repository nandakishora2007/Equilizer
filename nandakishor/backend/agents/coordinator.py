"""
Agent Coordinator.
Executes specialized detection agents asynchronously in parallel.
"""

import asyncio
from typing import Dict, Any, List

from backend.agents.face_agent import FaceAgent
from backend.agents.audio_agent import AudioAgent
from backend.agents.lipsync_agent import LipSyncAgent
from backend.agents.motion_agent import MotionAgent
from backend.agents.metadata_agent import MetadataAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.logger import logger


class AgentCoordinator:
    def __init__(self):
        self.face_agent = FaceAgent()
        self.audio_agent = AudioAgent()
        self.lipsync_agent = LipSyncAgent()
        self.motion_agent = MotionAgent()
        self.metadata_agent = MetadataAgent()
        self.fusion_agent = FusionAgent()

    async def _run_agent_async(self, agent, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a single agent analysis method in an async executor thread pool."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, agent.analyze, processed_data)
        except Exception as e:
            logger.error(f"Execution error in {agent.name}: {str(e)}")
            return agent.format_result(
                score=0.0,
                confidence=0.0,
                error=f"Uncaught agent exception: {str(e)}",
            )

    async def analyze_media(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs all analysis agents concurrently and merges results via FusionAgent.

        Args:
            processed_data: Payload containing frame paths, audio paths, and metadata.

        Returns:
            Final unified analysis report.
        """
        logger.info("[Coordinator] Dispatching agents for parallel execution...")

        agents = [
            self.face_agent,
            self.audio_agent,
            self.lipsync_agent,
            self.motion_agent,
            self.metadata_agent,
        ]

        # Execute all agents concurrently
        tasks = [self._run_agent_async(agent, processed_data) for agent in agents]
        agent_results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

        # Merge outputs using fusion agent
        final_report = self.fusion_agent.aggregate(agent_results)
        return final_report


if __name__ == "__main__":
    async def main():
        coordinator = AgentCoordinator()
        sample_data = {
            "file_path": None,
            "frame_paths": [],
            "audio_path": None,
            "has_audio": False,
        }
        report = await coordinator.analyze_media(sample_data)
        print("\n=== Final Pipeline Report ===")
        print(report)

    asyncio.run(main())