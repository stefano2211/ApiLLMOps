import os
import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from loguru import logger
from app.core.config import settings
from app.core.security import verify_api_key

router = APIRouter()

from app.persistence.storage import storage

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
        
    os.makedirs("/tmp/datalake", exist_ok=True)
    local_master_file = f"/tmp/datalake/{tenant_id}_master.jsonl"
    object_name = f"{tenant_id}_master.jsonl"
    bucket = settings.S3_BUCKET_DATALAKE
    
    try:
        # 1. Descargar histórico base desde S3 (evita sobreescrituras en blanco)
        storage.download_file(bucket, object_name, local_master_file)
        
        # 2. Append en disco local temporal en chunks (Evita OOM en archivos masivos)
        last_char = b''
        async with aiofiles.open(local_master_file, 'ab') as out_file:
            while True:
                chunk = await file.read(1024 * 1024)  # Chunks de 1MB
                if not chunk:
                    break
                await out_file.write(chunk)
                last_char = chunk[-1:]
            
            if last_char and last_char != b'\n':
                await out_file.write(b'\n')
                
        # 3. Empujar de vuelta al Data Lake (S3)
        storage.upload_file(bucket, object_name, local_master_file)
        
        # Limpiar
        if os.path.exists(local_master_file):
            os.remove(local_master_file)

        logger.info(f"Dataset securely appended to S3: {bucket}/{object_name}")
        return {"status": "success", "message": f"Data appended to S3 bucket: {bucket}/{object_name}"}
        
    except Exception as e:
        logger.error(f"Error appending dataset to MinIO: {e}")
        raise HTTPException(status_code=500, detail="Internal S3 Server Error")
