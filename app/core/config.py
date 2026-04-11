from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ApiLLMOps_Mothership"
    API_V1_STR: str = "/api/v1"

    # Security — set MOTHERSHIP_API_KEY env var in production; no insecure default enforced
    API_KEY: str = "default-mothership-secret-key"

    # CORS — list specific origins; "*" is incompatible with allow_credentials=True
    BACKEND_CORS_ORIGINS: list[str] = []

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Storage (Data Lake / S3)
    DATA_LAKE_PATH: str = "/tmp/datalake"
    MODELS_OUTPUT_PATH: str = "/tmp/models"

    # AWS S3 (MinIO)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_EXTERNAL_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_DATALAKE: str = "datalake"
    S3_BUCKET_DATALAKE_VL: str = "datalake-vl"
    S3_BUCKET_MODELS: str = "models"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
