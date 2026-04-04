import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ApiLLMOps_Mothership"
    API_V1_STR: str = "/api/v1"
    
    # Security
    API_KEY: str = os.getenv("MOTHERSHIP_API_KEY", "default-mothership-secret-key")
    
    # CORS
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Celery / Redis
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # Storage (Data Lake / S3)
    DATA_LAKE_PATH: str = os.getenv("DATA_LAKE_PATH", "/tmp/datalake")
    MODELS_OUTPUT_PATH: str = os.getenv("MODELS_OUTPUT_PATH", "/tmp/models")

    # AWS S3 (MinIO)
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_EXTERNAL_ENDPOINT: str = os.getenv("MINIO_EXTERNAL_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    S3_BUCKET_DATALAKE: str = os.getenv("S3_BUCKET_DATALAKE", "datalake")
    S3_BUCKET_DATALAKE_VL: str = os.getenv("S3_BUCKET_DATALAKE_VL", "datalake-vl")  # VL (screenshots + acciones)
    S3_BUCKET_MODELS: str = os.getenv("S3_BUCKET_MODELS", "models")

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
