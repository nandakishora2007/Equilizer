"""
Logger module providing consistent console and rotating file logging.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

from backend.core.constants import BASE_DIR

# =============================================================================
# Logging Configuration
# =============================================================================

# Directory where log files will be stored
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / "app.log"

# Log formatting
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Read log level from environment variable (defaults to DEBUG)
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()


def setup_logger(name: str = "deepfake_agentic") -> logging.Logger:
    """
    Configure and return a reusable logger.

    Features:
    - Console logging (INFO and above)
    - Rotating file logging (DEBUG and above)
    - Automatic log directory creation
    - Duplicate handler prevention
    - Log rotation (10 MB per file, 5 backups)
    """

    logger = logging.getLogger(name)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger

    # Ensure logger level is low enough to let DEBUG reach the file handler
    env_level = getattr(logging, LOG_LEVEL, logging.DEBUG)
    logger.setLevel(min(env_level, logging.DEBUG))

    # Formatter
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    # -------------------------------------------------------------------------
    # Console Handler (Prints clean output to terminal)
    # -------------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # -------------------------------------------------------------------------
    # Rotating File Handler (Captures deep debug details in log file)
    # -------------------------------------------------------------------------
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# Default Logger Instance
# =============================================================================

logger = setup_logger()