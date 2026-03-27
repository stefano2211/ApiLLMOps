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
        
    os.makedirs("/tmp/datalake", exist_ok=True)
    # Nombre de objeto dinámico: el nombre que viene del Edge (ej: aura_tenant_01_Sensor_Caldera.jsonl)
    object_name = file.filename
    local_master_file = f"/tmp/datalake/{object_name}"
    bucket = settings.S3_BUCKET_DATALAKE
    
    try:
        # 1. Descargar histórico específico del archivo si existe (Incremental Append)
        # Si no existe, storage no lanzará error o se manejará el flujo
        try:
            storage.download_file(bucket, object_name, local_master_file)
        except Exception:
            logger.info(f"New dataset file entry: {object_name}. Creating fresh.")
        
        # 2. Append en disco local temporal
        async with aiofiles.open(local_master_file, 'ab') as out_file:
            # Asegurar nueva línea al inicio si el archivo no está vacío
            if os.path.exists(local_master_file) and os.path.getsize(local_master_file) > 0:
                 await out_file.write(b'\n')
                 
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out_file.write(chunk)
                
        # 3. Re-subir al Bucket con el nombre específico de la herramienta
        storage.upload_file(bucket, object_name, local_master_file)
        
        # Limpiar
        if os.path.exists(local_master_file):
            os.remove(local_master_file)

        logger.info(f"Dataset securely appended to S3: {bucket}/{object_name}")
        return {"status": "success", "message": f"Data appended to S3 bucket: {bucket}/{object_name}"}
        
    except Exception as e:
        logger.error(f"Error appending dataset to MinIO: {e}")
        raise HTTPException(status_code=500, detail="Internal S3 Server Error")
