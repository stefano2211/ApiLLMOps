from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class TrainingJobRequest(BaseModel):
    tenant_id: str
    base_model: str = Field(..., description="HF model ID de la versión Qwen base", example="unsloth/qwen2.5-7b-bnb-4bit")
    epochs: int = 3
    webhook_url: Optional[HttpUrl] = None

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
