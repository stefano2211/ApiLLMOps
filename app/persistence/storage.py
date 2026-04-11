from minio import Minio
from minio.error import S3Error
from app.core.config import settings
from loguru import logger
import datetime
import os

class MinioManager:
    def __init__(self):
        # Internal client — used for all read/write operations (upload, download, list).
        # Uses MINIO_ENDPOINT (e.g. "minio:9000" inside Docker).
        internal_endpoint = settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        self.client = Minio(
            internal_endpoint,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_ENDPOINT.startswith("https://"),
        )

        # External client — used ONLY for generating presigned URLs that the Edge can reach.
        # Uses MINIO_EXTERNAL_ENDPOINT (e.g. "host.docker.internal:9002" or a public hostname).
        # Falls back to the internal client if no external endpoint is configured.
        external_endpoint = settings.MINIO_EXTERNAL_ENDPOINT.replace("http://", "").replace("https://", "")
        if external_endpoint and external_endpoint != internal_endpoint:
            self._presign_client = Minio(
                external_endpoint,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_EXTERNAL_ENDPOINT.startswith("https://"),
            )
        else:
            self._presign_client = self.client

    def init_buckets(self):
        """Creates buckets if they do not exist."""
        try:
            for bucket in [settings.S3_BUCKET_DATALAKE, settings.S3_BUCKET_DATALAKE_VL, settings.S3_BUCKET_MODELS]:
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
        """Returns a temporary download link for the edge device using the external endpoint."""
        try:
            self.client.stat_object(bucket, object_name)
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchObject"):
                logger.warning(f"[MinIO] Object not found, cannot generate presigned URL: {bucket}/{object_name}")
                return ""
            logger.error(f"[MinIO] stat_object error: {e}")
            return ""
        try:
            return self._presign_client.presigned_get_object(bucket, object_name, expires=expires)
        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            return ""

    def list_objects(self, bucket: str, prefix: str):
        """Lists objects in a bucket with a given prefix."""
        try:
            return self.client.list_objects(bucket, prefix=prefix, recursive=True)
        except S3Error as e:
            logger.error(f"Error listing objects in MinIO: {e}")
            return []

storage = MinioManager()
