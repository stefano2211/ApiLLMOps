from fastapi import APIRouter
from app.api.endpoints import datasets, training, models

api_router = APIRouter()

api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(training.router, prefix="/training", tags=["Training"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
