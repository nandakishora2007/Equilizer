"""
Utility functions for video frame extraction, face detection, and cropping using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from backend.core.logger import logger


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps_sample_rate: Optional[float] = 1.0,
    max_frames: int = 100,
) -> List[Path]:
    """
    Extracts frames from a video file at a target sampling rate (frames per second).

    Args:
        video_path: Path to input video file.
        output_dir: Directory where extracted frame images will be saved.
        fps_sample_rate: Frames to extract per second of video. If None, extracts every frame.
        max_frames: Safety limit to prevent extracting thousands of frames during hackathons.

    Returns:
        List of Paths to saved frame images (.jpg).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame step interval based on target fps_sample_rate
    if fps_sample_rate and fps_sample_rate > 0:
        step = max(1, int(original_fps / fps_sample_rate))
    else:
        step = 1

    extracted_frame_paths: List[Path] = []
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break

        if frame_count % step == 0:
            frame_filename = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            extracted_frame_paths.append(frame_filename)
            saved_count += 1

        frame_count += 1

    cap.release()
    logger.info(
        f"Extracted {saved_count} frames from {video_path.name} "
        f"(Total Frames: {total_frames}, Step: {step})"
    )
    return extracted_frame_paths


def crop_face(
    frame: np.ndarray,
    padding: float = 0.2,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Detects and crops the primary face from an image frame using OpenCV's Haar Cascade.

    Args:
        frame: OpenCV image array (BGR format).
        padding: Fractional padding around the detected face bounding box.

    Returns:
        Tuple of (cropped_face_array, bounding_box_coords (x, y, w, h)) or (None, None).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load default OpenCV Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None

    # Pick the largest detected face box by area
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Apply padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    img_h, img_w, _ = frame.shape
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face_crop = frame[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2 - x1, y2 - y1)