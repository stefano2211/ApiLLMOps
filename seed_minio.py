import json
import os
from minio import Minio
from loguru import logger

# Configuración manual para coincidir con docker-compose
MINIO_ENDPOINT = "localhost:9002" # Acceso desde fuera de Docker
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "datalake"
TENANT_ID = "aura_tenant_01"
OBJECT_NAME = f"{TENANT_ID}_master.jsonl"

def seed_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    # Crear bucket si no existe
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        logger.info(f"Bucket {BUCKET_NAME} creado.")

    # Generar data de prueba minimalista (ChatML format)
    dummy_data = [
        {"conversations": [
            {"from": "human", "value": "Hola, ¿quién eres?"},
            {"from": "gpt", "value": "Soy AURA, tu asistente industrial optimizado para el Edge."}
        ]},
        {"conversations": [
            {"from": "human", "value": "¿Qué puedes hacer?"},
            {"from": "gpt", "value": "Puedo analizar telemetría en tiempo real y detectar anomalías en la planta."}
        ]}
    ]

    # Guardar localmente
    local_path = "temp_dataset.jsonl"
    with open(local_path, "w", encoding="utf-8") as f:
        for entry in dummy_data:
            f.write(json.dumps(entry) + "\n")

    # Subir a MinIO
    client.fput_object(BUCKET_NAME, OBJECT_NAME, local_path)
    logger.info(f"Dataset de prueba subido a {BUCKET_NAME}/{OBJECT_NAME}")
    
    # Limpieza
    os.remove(local_path)

if __name__ == "__main__":
    try:
        seed_minio()
    except Exception as e:
        logger.error(f"Error subiendo seed data: {e}")
