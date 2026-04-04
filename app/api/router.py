from fastapi import APIRouter
from app.api.endpoints import datasets, training, models
from app.api.endpoints import vl_datasets, vl_training

api_router = APIRouter()

# ── Text pipeline (flujo existente) ──────────────────────────────────────────
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(training.router, prefix="/training", tags=["Training"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])

# ── VL pipeline (Macrohard — computer use) ───────────────────────────────────
api_router.include_router(vl_datasets.router, prefix="/vl", tags=["VL-Datasets"])
api_router.include_router(vl_training.router, prefix="/vl", tags=["VL-Training"])
