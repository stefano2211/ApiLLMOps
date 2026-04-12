from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class TrainingJobRequest(BaseModel):
    tenant_id: str
    base_model: str = Field(
        default="Qwen/Qwen3.5-2B",  # Debe coincidir exactamente con el modelo en vllm-sistema1
        description="HF model ID del modelo base. Debe ser el mismo que corre en vllm-sistema1."
    )
    epochs: int = 3
    webhook_url: Optional[HttpUrl] = None

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
