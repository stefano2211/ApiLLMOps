from fastapi import APIRouter
import platform
import os
from app.core.config import settings
from app.persistence.storage import storage

router = APIRouter()

@router.get("/{tenant_id}/latest/config")
async def get_latest_model(tenant_id: str):
    """
    Simula un Registry de modelos. Devuelve Presigned URLs auto-generadas 
    en S3 para que la fábrica asigne su ancho de banda en descargas directas.
    """
    bucket = settings.S3_BUCKET_MODELS
    gguf_name = f"{tenant_id}-v2.gguf"
    modelfile_name = f"{tenant_id}-v2.Modelfile"

    gguf_url = storage.get_presigned_url(bucket, gguf_name)
    modelfile_url = storage.get_presigned_url(bucket, modelfile_name)

    return {
        "tenant_id": tenant_id,
        "latest_tag": f"{tenant_id}-v2",
        "gguf_url": gguf_url,
        "modelfile_url": modelfile_url,
        "format": "GGUF",
        "quantization": "Q4_K_M"
    }
