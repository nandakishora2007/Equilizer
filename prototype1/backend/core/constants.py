"""
Global constants used throughout the application.

This file is supposed to grow with the future iterations.

"""

'''from pathlib import Path

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

API_PREFIX = "/api/v1"'''



"""
Global constants used throughout the Deepfake Agentic Detection Engine.

This module contains application-wide constants such as:
- Project paths
- Supported media formats
- Detection labels
- Agent names
- Agent fusion weights
- Risk levels
- Processing status
- Confidence thresholds
- Report keys
- AI model identifiers
- API configuration
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# =============================================================================
# Supported File Types
# =============================================================================

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
    ".flac",
    ".ogg",
}

# =============================================================================
# Media Types
# =============================================================================

VIDEO = "VIDEO"
AUDIO = "AUDIO"

# =============================================================================
# Detection Labels
# =============================================================================

REAL = "REAL"
DEEPFAKE = "DEEPFAKE"
UNKNOWN = "UNKNOWN"

# =============================================================================
# Risk Levels
# =============================================================================

LOW_RISK = "LOW"
MEDIUM_RISK = "MEDIUM"
HIGH_RISK = "HIGH"

# =============================================================================
# Processing Status
# =============================================================================

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

# =============================================================================
# Agent Names
# =============================================================================

COORDINATOR_AGENT = "Coordinator Agent"

FACE_AGENT = "Face Analysis Agent"
AUDIO_AGENT = "Audio Analysis Agent"
LIPSYNC_AGENT = "Lip Sync Agent"
MOTION_AGENT = "Motion Analysis Agent"
METADATA_AGENT = "Metadata Analysis Agent"

FUSION_AGENT = "Evidence Fusion Agent"

# =============================================================================
# Agent Fusion Weights
# =============================================================================
# These should sum to 1.0

AGENT_WEIGHTS = {
    FACE_AGENT: 0.35,
    AUDIO_AGENT: 0.25,
    LIPSYNC_AGENT: 0.20,
    MOTION_AGENT: 0.15,
    METADATA_AGENT: 0.05,
}

# =============================================================================
# AI Model Names
# =============================================================================
# These are logical identifiers. Replace with actual model names later.

FACE_MODEL = "FaceForensics++"
AUDIO_MODEL = "Wav2Vec2"
LIPSYNC_MODEL = "SyncNet"
MOTION_MODEL = "VideoMAE"
METADATA_MODEL = "FFprobe"

# =============================================================================
# Confidence Thresholds
# =============================================================================

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60
LOW_CONFIDENCE = 0.40

# =============================================================================
# Report Keys
# =============================================================================

REPORT_SUMMARY = "summary"
REPORT_EVIDENCE = "evidence"
REPORT_CONFIDENCE = "confidence"
REPORT_VERDICT = "verdict"
REPORT_RISK = "risk"
REPORT_AGENTS = "agents"
REPORT_TIMESTAMP = "timestamp"

# =============================================================================
# Default Report Settings
# =============================================================================

DEFAULT_REPORT_EXTENSION = ".json"

# =============================================================================
# Logging
# =============================================================================

LOG_FILE_NAME = "app.log"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# =============================================================================
# API
# =============================================================================

API_PREFIX = "/api/v1"