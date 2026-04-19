"""
VL Training Endpoint
====================
Dispara el job de entrenamiento VL (computer use) en el Celery Worker GPU.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from pydantic import BaseModel
from app.core.security import verify_api_key
from app.domain.services.vl_trainer import start_vl_finetuning_task
router = APIRouter()


class VLTrainingJobRequest(BaseModel):
    tenant_id: str = "aura_tenant_01"
    base_model: str = "Qwen/Qwen3.5-2B"  # Debe coincidir exactamente con el modelo en vllm-sistema1
    vl_epochs: int = 2        # Épocas de entrenamiento con datos de computer use (screenshots + acciones)
    text_epochs: int = 0      # Mantenido por compatibilidad — ignorado (NO hay mezcla de texto en VL trainer)
    webhook_url: Optional[str] = None



class VLTrainingJobResponse(BaseModel):
    job_id: str
    status: str
    detail: str = ""


@router.post("/training/job", response_model=VLTrainingJobResponse)
async def trigger_vl_training(
    req: VLTrainingJobRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Encola el pipeline VL en el Celery Worker GPU.

    El pipeline ejecuta entrenamiento de computer use con datos de screenshots + acciones
    (formato: messages con PIL images + respuestas JSON de acción).
    
    Al finalizar, dispara el webhook OTA al Edge con payload:
      {"model_tag": "...-vl", "model_type": "vision"}
    El Edge descarga el tar.gz del adaptador LoRA safetensors y lo inyecta en vLLM.
    """
    logger.info(f"[VL Training] Job recibido para tenant: {req.tenant_id}")
    try:
        task = start_vl_finetuning_task.delay(
            tenant_id=req.tenant_id,
            base_model=req.base_model,
            vl_epochs=req.vl_epochs,
            text_epochs=req.text_epochs,
            webhook_url=str(req.webhook_url) if req.webhook_url else None,
        )
        logger.info(f"[VL Training] Job encolado: {task.id}")
        return VLTrainingJobResponse(
            job_id=task.id,
            status="queued",
            detail=f"Pipeline VL iniciado para {req.tenant_id}",
        )
    except Exception as e:
        logger.error(f"[VL Training] Error al encolar Celery: {e}")
        raise HTTPException(status_code=500, detail="Error al encolar el job VL")


@router.get("/training/job/{job_id}/status")
async def get_vl_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Consulta el estado de un job VL en Celery."""
    result = start_vl_finetuning_task.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


