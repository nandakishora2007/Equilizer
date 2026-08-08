from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    # Application
    PROJECT_NAME: str = "Deepfake Agentic Detection Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Upload settings
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: set[str] = {
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".mp3",
    }

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


# Global settings instance
settings = Settings()