from fastapi import APIRouter, Depends
import platform
import os
from app.core.config import settings
from app.core.security import verify_api_key
from app.persistence.storage import storage

router = APIRouter()

@router.get("/{tenant_id}/latest/config")
async def get_latest_model(
    tenant_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Simula un Registry de modelos. Devuelve Presigned URLs auto-generadas 
    en S3 para que la fábrica asigne su ancho de banda en descargas directas.
    Requiere API key para prevenir descarga no autorizada de modelos.
    """
    bucket = settings.S3_BUCKET_MODELS
    tar_name = f"{tenant_id}-v2-lora.tar.gz"

    tar_url = storage.get_presigned_url(bucket, tar_name)

    return {
        "tenant_id": tenant_id,
        "latest_tag": f"{tenant_id}-v2",
        "lora_url": tar_url,
        "format": "Safetensors-LoRA",
        "quantization": "BF16"
    }
