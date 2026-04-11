from fastapi import APIRouter, Depends, HTTPException
from app.core.config import settings
from app.core.security import verify_api_key
from app.persistence.storage import storage

router = APIRouter()


def _find_latest_lora(bucket: str, tenant_id: str, suffix: str = "-lora.tar.gz") -> str | None:
    """
    Lists all tar.gz LoRA artifacts for a tenant and returns the most recently
    uploaded object name, or None if no artifact exists yet.
    """
    prefix = f"{tenant_id}-"
    objects = list(storage.list_objects(bucket, prefix=prefix))
    candidates = [o for o in objects if o.object_name.endswith(suffix)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda o: o.last_modified)
    return latest.object_name


@router.get("/{tenant_id}/latest/config")
async def get_latest_model(
    tenant_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Model registry endpoint. Returns a presigned URL for the latest trained
    LoRA adapter so Edge devices can download it directly from MinIO.
    """
    bucket = settings.S3_BUCKET_MODELS
    tar_name = _find_latest_lora(bucket, tenant_id)

    if not tar_name:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for tenant '{tenant_id}'. Run a training job first.",
        )

    tar_url = storage.get_presigned_url(bucket, tar_name)
    if not tar_url:
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL")

    tag = tar_name.replace("-lora.tar.gz", "").replace(f"{tenant_id}-", "", 1)
    return {
        "tenant_id": tenant_id,
        "latest_tag": f"{tenant_id}-{tag}",
        "lora_url": tar_url,
        "format": "Safetensors-LoRA",
        "quantization": "BF16",
    }
