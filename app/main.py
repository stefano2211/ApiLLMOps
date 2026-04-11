import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import settings
from app.persistence.storage import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_buckets()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Mothership API for Industrial LLM Fine-Tuning (Edge-to-Cloud)",
    lifespan=lifespan,
)

# CORS — only enabled when specific origins are configured.
# allow_credentials=True is NOT compatible with allow_origins=["*"] per the CORS spec.
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/healthz")
def health_check():
    return {"status": "ok", "service": "apillmops-mothership"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
