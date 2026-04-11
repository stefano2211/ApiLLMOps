import os
import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from loguru import logger
from app.core.config import settings
from app.core.security import verify_api_key
from app.persistence.storage import storage

router = APIRouter()

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...), 
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """
    Subida y anexado de Data Lake (Replay Data) hacia el Bucket MinIO S3.
    Descarga archivo temporal, anexa, y re-sube como objeto atómico.
    """
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are supported")
        
    import uuid
    os.makedirs(settings.DATA_LAKE_PATH, exist_ok=True)
    base_name = file.filename
    safe_uuid = str(uuid.uuid4())[:8]
    if base_name.endswith(".jsonl"):
        object_name = base_name.replace(".jsonl", f"_{safe_uuid}.jsonl")
    else:
        object_name = f"{base_name}_{safe_uuid}.jsonl"

    local_chunk_file = os.path.join(settings.DATA_LAKE_PATH, object_name)
    bucket = settings.S3_BUCKET_DATALAKE
    
    try:
        # Descarga el stream entrante en el chunk nuevo local
        async with aiofiles.open(local_chunk_file, 'wb') as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out_file.write(chunk)

        # Re-subir al Bucket como partición única independiente
        storage.upload_file(bucket, object_name, local_chunk_file)

        logger.info(f"Dataset securely partitioned to S3: {bucket}/{object_name}")
        return {"status": "success", "message": f"Data chunk uploaded to S3 bucket: {bucket}/{object_name}"}

    except Exception as e:
        logger.error(f"Error appending dataset to MinIO: {e}")
        raise HTTPException(status_code=500, detail="Internal S3 Server Error")

    finally:
        if os.path.exists(local_chunk_file):
            os.remove(local_chunk_file)
