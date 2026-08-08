"""
Fusion Agent: Aggregates individual agent scores into a final verdict.
"""

from typing import Dict, Any
from backend.agents.base_agent import BaseAgent
from backend.core.logger import logger

FUSION_AGENT = "Fusion Agent"


class FusionAgent(BaseAgent):
    def __init__(self, model_path: Any = None):
        super().__init__(name=FUSION_AGENT)

    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregates individual scores from face, video, audio, and metadata agents.
        """
        try:
            agent_results = processed_data.get("agent_results", {})

            scores = []
            weights = {"face": 0.40, "video": 0.30, "audio": 0.20, "metadata": 0.10}
            weighted_score = 0.0
            total_weight = 0.0

            for agent_key, weight in weights.items():
                if agent_key in agent_results:
                    res = agent_results[agent_key]
                    score = res.get("score", 0.5)
                    weighted_score += score * weight
                    total_weight += weight
                    scores.append(score)

            overall_score = round(weighted_score / total_weight, 4) if total_weight > 0 else 0.50

            # Determine explicit text verdict
            if overall_score >= 0.65:
                verdict = "DEEPFAKE"
            elif overall_score <= 0.35:
                verdict = "REAL"
            else:
                verdict = "SUSPECTED / INCONCLUSIVE"

            confidence = round(min(0.95, 0.60 + (len(scores) * 0.08)), 2)

            return {
                "verdict": verdict,
                "overall_score": overall_score,
                "confidence": confidence,
                "agent_results": agent_results,
                "details": [
                    f"Aggregated evaluation across {len(scores)} analysis modalities.",
                    f"Weighted Deepfake Risk Score: {overall_score * 100:.1f}%",
                    f"Final Verdict Determination: {verdict}"
                ]
            }

        except Exception as e:
            logger.error(f"[{self.name}] Fusion analysis error: {e}")
            return {
                "verdict": "DEEPFAKE DETECTED (FALLBACK)",
                "overall_score": 0.78,
                "confidence": 0.75,
                "agent_results": processed_data.get("agent_results", {}),
                "details": [f"Fusion synthesis fallback active: {str(e)}"]
            }