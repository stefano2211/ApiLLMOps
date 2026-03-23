from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from app.core.security import verify_api_key
from app.domain.schemas.training import TrainingJobRequest, TrainingJobResponse
from app.domain.services.unsloth_trainer import start_finetuning_task

router = APIRouter()

@router.post("/job", response_model=TrainingJobResponse)
async def trigger_training(
    req: TrainingJobRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Gatilla un trabajo de entrenamiento en background asíncrono.
    """
    logger.info(f"Received training request for tenant: {req.tenant_id}")
    
    # 1. Encola la tarea en Celery
    try:
        task = start_finetuning_task.delay(
            tenant_id=req.tenant_id,
            base_model=req.base_model,
            epochs=req.epochs,
            webhook_url=str(req.webhook_url) if req.webhook_url else None
        )
        return {"job_id": task.id, "status": "queued"}
    except Exception as e:
        logger.error(f"Celery enqueue failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue training job")

@router.get("/job/{job_id}/status")
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Consulta el estado de una tarea de Celery.
    """
    result = start_finetuning_task.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }
