"""
Fusion Agent.
Aggregates and weighs individual agent scores to generate a unified deepfake verdict.
"""

from typing import Dict, Any, List
from backend.core.constants import REAL, DEEPFAKE, UNKNOWN, AGENT_WEIGHTS
from backend.core.logger import logger


class FusionAgent:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or AGENT_WEIGHTS

    def aggregate(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines outputs from multiple detection agents using weighted confidence.

        Args:
            agent_results: List of standard agent response dictionaries.

        Returns:
            Unified payload containing aggregated score, verdict, overall confidence,
            and compiled evidence details.
        """
        if not agent_results:
            return {
                "final_score": 0.0,
                "verdict": UNKNOWN,
                "confidence": 0.0,
                "summary": ["No agent outputs available for fusion analysis."],
                "agent_breakdown": [],
            }

        total_weight = 0.0
        weighted_score_sum = 0.0
        confidence_sum = 0.0
        valid_agents = 0
        compiled_details = []

        for result in agent_results:
            agent_name = result.get("agent_name", "Unknown")
            error = result.get("error")

            if error:
                logger.warning(f"[Fusion] Skipping {agent_name} due to error: {error}")
                continue

            score = result.get("score", 0.0)
            confidence = result.get("confidence", 0.0)
            details = result.get("details", [])

            # Fetch agent weight (default to 1.0 if unspecified)
            weight = self.weights.get(agent_name, 1.0)

            # Accumulate weighted score based on confidence
            effective_weight = weight * confidence
            weighted_score_sum += score * effective_weight
            total_weight += effective_weight

            confidence_sum += confidence
            valid_agents += 1

            for detail in details:
                compiled_details.append(f"[{agent_name}] {detail}")

        if total_weight == 0.0 or valid_agents == 0:
            return {
                "final_score": 0.0,
                "verdict": UNKNOWN,
                "confidence": 0.0,
                "summary": ["Insufficient valid agent data to compute a final verdict."],
                "agent_breakdown": agent_results,
            }

        final_score = round(weighted_score_sum / total_weight, 4)
        avg_confidence = round(confidence_sum / valid_agents, 4)

        if final_score >= 0.5:
            verdict = DEEPFAKE
            summary_statement = f"Media classified as DEEPFAKE with {int(final_score * 100)}% likelihood."
        else:
            verdict = REAL
            summary_statement = f"Media classified as REAL with {int((1 - final_score) * 100)}% authenticity likelihood."

        logger.info(f"[Fusion] Aggregated Verdict: {verdict} (Score: {final_score}, Conf: {avg_confidence})")

        return {
            "final_score": final_score,
            "verdict": verdict,
            "confidence": avg_confidence,
            "summary": [summary_statement] + compiled_details,
            "agent_breakdown": agent_results,
        }


if __name__ == "__main__":
    fusion = FusionAgent()
    mock_results = [
        {"agent_name": "Face Analysis Agent", "score": 0.85, "confidence": 0.90, "details": ["Face boundary blur"], "error": None},
        {"agent_name": "Audio Analysis Agent", "score": 0.20, "confidence": 0.80, "details": ["Natural audio"], "error": None},
    ]
    print("Fusion Output:", fusion.aggregate(mock_results))