from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker concurrency behavior
    worker_concurrency=1, # Important: 1 GPU = 1 concurrent training job max
    worker_prefetch_multiplier=1, # Prevent a worker from holding multiple heavy jobs
    task_track_started=True,
    task_time_limit=3600 * 5,  # Max 5 hours for training
    imports=["app.domain.services.unsloth_trainer"],
)
