from app.core.celery_app import celery_app

# This file acts as the entrypoint for the Celery Worker.
# Run it with: celery -A worker.celery_app worker --loglevel=info --concurrency=1 -O fair

if __name__ == "__main__":
    celery_app.start()
