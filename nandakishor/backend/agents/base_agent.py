"""
Abstract base class for all deepfake detection agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from backend.core.constants import REAL, DEEPFAKE, UNKNOWN
from backend.core.logger import logger


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized agent: {self.name}")

    @abstractmethod
    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def format_result(
        self,
        score: float,
        confidence: float,
        details: List[str] = None,
        error: str = None,
    ) -> Dict[str, Any]:
        details = details or []

        if error:
            verdict = UNKNOWN
        elif score >= 0.5:
            verdict = DEEPFAKE
        else:
            verdict = REAL

        return {
            "agent_name": self.name,
            "score": round(float(score), 4),
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "details": details,
            "error": error,
        }