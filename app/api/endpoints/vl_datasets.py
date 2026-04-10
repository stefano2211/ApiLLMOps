"""
VL Datasets Endpoint
====================
Maneja la subida de datasets Vision-Language (screenshots + acciones JSON)
al bucket MinIO `datalake-vl`, separado del datalake de texto.

Formato JSONL esperado (compatible con FastVisionModel):
  {
    "messages": [
      {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "{\"type\":\"click\",...}"}]}
    ],
    "images": ["<base64_png_string>"]
  }
"""
import os
import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from loguru import logger
from app.core.config import settings
from app.core.security import verify_api_key
from app.persistence.storage import storage

router = APIRouter()


@router.post("/upload")
async def upload_vl_dataset(
    file: UploadFile = File(...),
    tenant_id: str = "default",
    tool_name: str = "computer_use",
    api_key: str = Depends(verify_api_key),
):
    """
    Subida incremental de datasets Vision-Language al bucket MinIO `datalake-vl`.

    - El nombre del objeto en MinIO será: `{tenant_id}_vl_{tool_name}.jsonl`
    - Los datos se ANEXAN al archivo existente (modo append), igual que el datalake de texto.
    - Acepta archivos .jsonl con pares (screenshot_base64, action_json).
    """
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .jsonl")

    os.makedirs("/tmp/datalake-vl", exist_ok=True)
    object_name = f"{tenant_id}_vl_{tool_name}.jsonl"
    local_path = f"/tmp/datalake-vl/{object_name}"
    bucket = settings.S3_BUCKET_DATALAKE_VL

    try:
        # Descargar histórico si existe para hacer append.
        # download_file() returns False (not raises) for NoSuchKey, so any exception here
        # is a real infrastructure error (network, auth, bucket missing) and must propagate.
        found = storage.download_file(bucket, object_name, local_path)
        if found:
            logger.info(f"[VL Dataset] Histórico descargado: {object_name}")
        else:
            logger.info(f"[VL Dataset] Nuevo archivo VL: {object_name}")

        # Append en disco local
        async with aiofiles.open(local_path, "ab") as out_file:
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                await out_file.write(b"\n")
            while True:
                chunk = await file.read(1024 * 1024)  # chunks de 1MB
                if not chunk:
                    break
                await out_file.write(chunk)

        # Re-subir al bucket
        storage.upload_file(bucket, object_name, local_path)

        logger.info(f"[VL Dataset] ✅ Dataset VL subido: {bucket}/{object_name}")
        return {
            "status": "success",
            "message": f"Dataset VL anexado a MinIO: {bucket}/{object_name}",
            "object": object_name,
        }

    except Exception as e:
        logger.error(f"[VL Dataset] Error subiendo a MinIO: {e}")
        raise HTTPException(status_code=500, detail="Error interno al subir dataset VL")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
