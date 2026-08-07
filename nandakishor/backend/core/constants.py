"""
Global constants used throughout the application.

This file is supposed to grow with the future iterations.

"""

from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"

# ============================================================
# Supported File Types
# ============================================================

SUPPORTED_VIDEO_FORMATS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

SUPPORTED_AUDIO_FORMATS = {
    ".wav",
    ".mp3",
    ".aac",
}

# ============================================================
# Detection Labels
# ============================================================

REAL = "REAL"
DEEPFAKE = "DEEPFAKE"
UNKNOWN = "UNKNOWN"

# ============================================================
# Risk Levels
# ============================================================

LOW_RISK = "LOW"
MEDIUM_RISK = "MEDIUM"
HIGH_RISK = "HIGH"

# ============================================================
# Agent Names
# ============================================================

COORDINATOR_AGENT = "Coordinator Agent"
FACE_AGENT = "Face Analysis Agent"
AUDIO_AGENT = "Audio Analysis Agent"
LIPSYNC_AGENT = "Lip Sync Agent"
MOTION_AGENT = "Motion Analysis Agent"
METADATA_AGENT = "Metadata Analysis Agent"
FUSION_AGENT = "Evidence Fusion Agent"

# ============================================================
# Processing Status
# ============================================================

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

# ============================================================
# Default Confidence Thresholds
# ============================================================

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60
LOW_CONFIDENCE = 0.40

# ============================================================
# Report Keys
# ============================================================

REPORT_SUMMARY = "summary"
REPORT_EVIDENCE = "evidence"
REPORT_CONFIDENCE = "confidence"
REPORT_VERDICT = "verdict"

# ============================================================
# API
# ============================================================

API_PREFIX = "/api/v1"