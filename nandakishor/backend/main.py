"""
Main FastAPI Application Entrypoint for Equilizer Deepfake Detection Service.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.analyze import router as analyze_router
from backend.core.logger import logger

app = FastAPI(
    title="Equilizer - Multi-Agent Deepfake Detection API",
    version="1.0.0",
    description="Backend API for multi-modal deepfake detection using coordinated AI agents.",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust during production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(analyze_router)


@app.get("/")
def health_check():
    return {
        "status": "ONLINE",
        "service": "Equilizer Multi-Agent Backend",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Equilizer FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)