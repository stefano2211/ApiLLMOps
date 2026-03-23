from minio import Minio
from minio.error import S3Error
from app.core.config import settings
from loguru import logger
import datetime
import os

class MinioManager:
    def __init__(self):
        # Remove http/https from endpoint for the SDK client
        endpoint = settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        self.client = Minio(
            endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_ENDPOINT.startswith("https://")
        )

    def init_buckets(self):
        """Creates buckets if they do not exist."""
        try:
            for bucket in [settings.S3_BUCKET_DATALAKE, settings.S3_BUCKET_MODELS]:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
        except S3Error as e:
            logger.error(f"MinIO bucket error: {e}")

    def upload_file(self, bucket: str, object_name: str, file_path: str):
        """Uploads a local file to S3."""
        try:
            self.client.fput_object(bucket, object_name, file_path)
            logger.info(f"Uploaded {file_path} to {bucket}/{object_name}")
        except S3Error as e:
            logger.error(f"Error uploading to MinIO: {e}")
            raise e

    def download_file(self, bucket: str, object_name: str, file_path: str):
        """Downloads an object from S3 to a local file. Returns bool indicating if file existed."""
        try:
            self.client.fget_object(bucket, object_name, file_path)
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            logger.error(f"Error downloading from MinIO: {e}")
            raise e
        return True

    def get_presigned_url(self, bucket: str, object_name: str, expires=datetime.timedelta(hours=2)) -> str:
        """Returns a temporary download link for the edge device."""
        try:
            # Genera URL con el host interno usado por el Container (ej. minio:9000)
            url = self.client.presigned_get_object(bucket, object_name, expires=expires)
            
            # Auto-sustituye minio:9000 por localhost:9000 (o el dominio externo) 
            # para que el Edge Node físico pueda resolverlo desde afuera de Docker
            if settings.MINIO_ENDPOINT and settings.MINIO_EXTERNAL_ENDPOINT:
                url = url.replace(settings.MINIO_ENDPOINT, settings.MINIO_EXTERNAL_ENDPOINT)
                
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            return ""

storage = MinioManager()
