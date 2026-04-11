"""
VL Training Endpoint
====================
Dispara el job de entrenamiento VL unificado (2 fases) en el Celery Worker.
También expone el endpoint para consultar el modelo VL más reciente (presigned URLs).
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from pydantic import BaseModel, HttpUrl
from app.core.config import settings
from app.core.security import verify_api_key
from app.persistence.storage import storage
from app.domain.services.vl_trainer import start_vl_finetuning_task
from app.api.endpoints.models import _find_latest_lora

router = APIRouter()


class VLTrainingJobRequest(BaseModel):
    tenant_id: str = "aura_tenant_01"
    base_model: str = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"  # 3B: óptimo para GUI grounding en edge
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


@router.get("/models/{tenant_id}/vl/config")
async def get_vl_model_config(
    tenant_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Devuelve la presigned URL del adaptador LoRA VL del tenant.
    El Edge usa esta URL para OTA: descarga el tar.gz, extrae los pesos a /loras/
    y llama a POST /v1/load_lora_adapter en vLLM.
    """
    bucket = settings.S3_BUCKET_MODELS
    tar_name = _find_latest_lora(bucket, tenant_id, suffix="-vl-lora.tar.gz")

    if not tar_name:
        raise HTTPException(
            status_code=404,
            detail=f"No VL model found for tenant '{tenant_id}'. Run a VL training job first.",
        )

    lora_url = storage.get_presigned_url(bucket, tar_name)
    if not lora_url:
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL")

    return {
        "tenant_id": tenant_id,
        "model_type": "vision",
        "latest_tag": f"{tenant_id}-vl",
        "lora_url": lora_url,
        "format": "safetensors",
        "base_architecture": "Qwen2.5-VL",
    }
