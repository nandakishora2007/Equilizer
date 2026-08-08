"""
API Route for media upload and deepfake analysis execution.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.utils.file_utils import save_uploaded_stream
from backend.core.logger import logger

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/")
async def analyze_media(file: UploadFile = File(...)):
    """
    Accepts video/audio files via multipart/form-data, streams them to disk,
    and initializes processing pipeline.
    """
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
        )