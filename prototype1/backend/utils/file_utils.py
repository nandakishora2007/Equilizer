"""
Utility functions for safe file handling, dynamic upload directory generation,
extension validation, and temporary file management.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from backend.core.constants import (
    REPORTS_DIR,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    UPLOADS_DIR,
)
from backend.core.logger import logger


# =============================================================================
# Helper Functions
# =============================================================================

def generate_job_id() -> str:
    """
    Generate a unique job ID for execution tracking and folder isolation.
    """
    return str(uuid.uuid4())


def get_file_extension(filename: str) -> str:
    """
    Extract and normalize the file extension.

    Example:
        video.MP4 -> .mp4
    """
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """
    Check whether the file is a supported media type.
    """
    extension = get_file_extension(filename)
    return (
        extension in SUPPORTED_VIDEO_FORMATS
        or extension in SUPPORTED_AUDIO_FORMATS
    )


def is_supported_video(filename: str) -> bool:
    """
    Check whether the file is a supported video format.
    """
    return get_file_extension(filename) in SUPPORTED_VIDEO_FORMATS


def is_supported_audio(filename: str) -> bool:
    """
    Check whether the file is a supported audio format.
    """
    return get_file_extension(filename) in SUPPORTED_AUDIO_FORMATS


# =============================================================================
# File Management
# =============================================================================

def create_job_directory(job_id: Optional[str] = None) -> tuple[str, Path]:
    """
    Create an isolated upload directory for a processing job.

    Args:
        job_id: Existing job ID. If None, a new UUID is generated.

    Returns:
        (job_id, job_directory_path)
    """
    if job_id is None:
        job_id = generate_job_id()

    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Created job directory: {job_dir}")

    return job_id, job_dir


def save_uploaded_file(
    file_bytes: bytes,
    original_filename: str,
    job_id: Optional[str] = None,
) -> tuple[str, Path]:
    """
    Save uploaded media into an isolated job directory using raw bytes.

    Args:
        file_bytes: Binary contents of the uploaded file.
        original_filename: Original filename from the client.
        job_id: Existing job ID (optional).

    Returns:
        (job_id, saved_file_path)

    Raises:
        ValueError: If the upload is empty or the file extension is unsupported.
    """

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if not is_supported_file(original_filename):
        raise ValueError(
            f"Unsupported file type: {get_file_extension(original_filename)}"
        )

    job_id, job_dir = create_job_directory(job_id)

    # Store using a fixed internal filename
    extension = get_file_extension(original_filename)
    saved_path = job_dir / f"source_media{extension}"

    try:
        saved_path.write_bytes(file_bytes)

        logger.info(
            f"Upload saved successfully | "
            f"Job ID: {job_id} | "
            f"Path: {saved_path}"
        )

    except Exception:
        logger.exception(
            f"Failed to save uploaded file for Job ID: {job_id}"
        )
        raise

    return job_id, saved_path


def save_uploaded_stream(
    upload_file: UploadFile,
    job_id: Optional[str] = None,
) -> tuple[str, Path]:
    """
    Streams large video/audio files directly from FastAPI's UploadFile
    to disk in chunks, keeping memory footprint minimal.
    """
    filename = upload_file.filename or "unknown_file"

    if not is_supported_file(filename):
        raise ValueError(f"Unsupported file type: {get_file_extension(filename)}")

    job_id, job_dir = create_job_directory(job_id)
    extension = get_file_extension(filename)
    saved_path = job_dir / f"source_media{extension}"

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        logger.info(
            f"Streamed upload saved | Job ID: {job_id} | Path: {saved_path}"
        )
    except Exception:
        logger.exception(f"Failed to stream uploaded file for Job ID: {job_id}")
        raise

    return job_id, saved_path


def cleanup_job_directory(job_id: str) -> bool:
    """
    Remove an isolated processing directory and all generated files.

    Returns:
        True if deleted successfully, otherwise False.
    """
    job_dir = UPLOADS_DIR / job_id

    if not job_dir.exists():
        logger.warning(f"Job directory does not exist: {job_dir}")
        return False

    try:
        shutil.rmtree(job_dir)
        logger.info(f"Cleaned up job directory: {job_dir}")
        return True

    except Exception:
        logger.exception(f"Failed to clean up job directory: {job_dir}")
        return False


def get_report_file_path(
    job_id: str,
    extension: str = ".json",
) -> Path:
    """
    Return the output report path for a given job.

    Example:
        reports/<job_id>_report.json
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    return REPORTS_DIR / f"{job_id}_report{extension}"