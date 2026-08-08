"""
API Route for media upload and deepfake analysis execution.


from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.utils.file_utils import save_uploaded_stream
from backend.core.logger import logger

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/")
async def analyze_media(file: UploadFile = File(...)):
    
    Accepts video/audio files via multipart/form-data, streams them to disk,
    and initializes processing pipeline.
    
    try:
        # Call the streaming utility directly from file_utils.py
        job_id, saved_path = save_uploaded_stream(upload_file=file)

        return {
            "status": "SUCCESS",
            "job_id": job_id,
            "filename": file.filename,
            "saved_path": str(saved_path),
            "message": "File received and saved successfully."
        }

    except ValueError as e:
        logger.warning(f"Invalid upload attempt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unexpected error during file upload processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while saving the uploaded file."
        )"""










"""
API Route for media upload, processing, multi-agent analysis, and report generation.
"""

import json
from fastapi import APIRouter, File, UploadFile, HTTPException, status, BackgroundTasks
from backend.utils.file_utils import save_uploaded_stream, cleanup_job_directory, get_report_file_path
from backend.services.video_processor import VideoProcessor
from backend.agents.coordinator import AgentCoordinator
from backend.core.logger import logger

router = APIRouter(prefix="/analyze", tags=["Analyze"])

# Initialize single instances of processor and coordinator
video_processor = VideoProcessor(fps_sample_rate=1.0, max_frames=50)
coordinator = AgentCoordinator()


@router.post("/")
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts media file upload, extracts features, executes parallel agent analysis,
    and returns a unified deepfake report.
    """
    job_id = None
    try:
        # 1. Memory-safe streaming save to isolated disk folder
        job_id, saved_media_path = save_uploaded_stream(upload_file=file)
        job_dir = saved_media_path.parent

        # 2. Extract frame sequences and audio track
        processed_data = video_processor.process(
            media_path=saved_media_path,
            job_dir=job_dir,
        )

        # Include raw file_path for MetadataAgent
        processed_data["file_path"] = str(saved_media_path)

        # 3. Execute parallel multi-agent analysis
        report = await coordinator.analyze_media(processed_data)

        # 4. Attach job metadata to final payload
        full_payload = {
            "job_id": job_id,
            "filename": file.filename,
            "analysis": report,
        }

        # 5. Persist JSON report to reports/ directory
        report_path = get_report_file_path(job_id)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, indent=2)

        logger.info(f"Report persisted successfully: {report_path}")

        # Schedule temporary uploads folder cleanup in background
        background_tasks.add_task(cleanup_job_directory, job_id)

        return full_payload

    except ValueError as e:
        logger.warning(f"Validation error during upload: {str(e)}")
        if job_id:
            cleanup_job_directory(job_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error during media analysis pipeline execution.")
        if job_id:
            cleanup_job_directory(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the uploaded media file.",
        )