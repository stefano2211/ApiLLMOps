"""
VL Fine-Tuning Task (Celery Worker) — Computer Use Agent
==========================================================
Entrena el modelo Qwen2.5-VL-3B EXCLUSIVAMENTE con datos de Computer Use:
  - Screenshots de interfaces (SAP GUI, email, apps industriales)
  - Acciones JSON correspondientes (click, type, press, scroll)

Este modelo es el "Digital Optimus Local" del sistema Macrohard industrial.
NO mezcla datos de texto industrial — eso lo maneja el FastLanguageModel
en start_finetuning_task (pipeline de texto independiente).

Flujo de 4 pasos:
  1. Descarga datasets VL (screenshots + acciones) de MinIO datalake-vl
  2. Carga FastVisionModel (Qwen2.5-VL-3B) en 4-bit con LoRA
  3. Entrena con SFTConfig + UnslothVisionDataCollator
  4. Export GGUF + mmproj → MinIO → Webhook OTA al Edge
"""

from celery import shared_task
from loguru import logger
import time
import requests
import os
import gc
import json
import threading
from app.core.config import settings


@shared_task(bind=True)
def start_vl_finetuning_task(
    self,
    tenant_id: str,
    base_model: str,
    vl_epochs: int,
    text_epochs: int,   # mantenido por compatibilidad de API, ignorado en este trainer
    webhook_url: str,
):
    """
    Pipeline VL de 4 pasos (Computer Use puro — sin mezcla de texto):
      1. Agrega datasets VL de MinIO datalake-vl
      2. Carga FastVisionModel en 4-bit con LoRA
      3. Entrena con datos de screenshots + acciones
      4. Export GGUF + mmproj → MinIO → Webhook OTA
    """
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    logger.info(
        f"[VL-MLOps] 🚀 Iniciando Pipeline VL (Computer Use) para {tenant_id} | base={base_model}"
    )

    # ── Imports críticos GPU ─────────────────────────────────────────────────
    try:
        import torch
        import torch._dynamo
        try:
            import torch._inductor.config
        except Exception:
            pass

        from unsloth import FastVisionModel, is_bfloat16_supported
        from unsloth.trainer import UnslothVisionDataCollator
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset as HFDataset
        from PIL import Image as PILImage
        import io
        import base64

    except ImportError as e:
        logger.error(f"[VL-MLOps] FATAL: Librerías GPU no disponibles: {e}")
        return {"status": "error", "detail": f"Missing GPU libraries: {e}"}

    from app.persistence.storage import storage

    bucket_vl = settings.S3_BUCKET_DATALAKE_VL
    bucket_models = settings.S3_BUCKET_MODELS
    local_vl_path = f"/tmp/{tenant_id}_vl_master.jsonl"

    model = None
    tokenizer = None
    trainer = None

    try:
        # ── PASO 1: Agregar datasets VL (screenshots + acciones) ────────────
        logger.info(f"--> [Paso 1] Descargando datasets VL de {bucket_vl} para {tenant_id}...")
        has_vl_data = _aggregate_vl_datasets(storage, bucket_vl, tenant_id, local_vl_path)

        if not has_vl_data:
            raise FileNotFoundError(
                f"No se encontraron datasets VL para {tenant_id} en {bucket_vl}. "
                f"Sube datos primero via POST /api/v1/vl/upload"
            )

        # ── PASO 2: Cargar FastVisionModel en 4-bit ──────────────────────────
        logger.info(f"--> [Paso 2] Cargando FastVisionModel {base_model} en 4-bit...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=base_model,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        model = FastVisionModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            finetune_vision_layers=True,       # Fine-tunea el encoder visual
            finetune_language_layers=True,     # Fine-tunea el LLM para output JSON
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )

        # ── PASO 3: Entrenamiento VL (Computer Use puro) ─────────────────────
        logger.info(f"--> [Paso 3] Cargando dataset VL desde {local_vl_path}...")
        vl_dataset = _load_vl_dataset(local_vl_path)
        logger.info(f"    Dataset cargado: {len(vl_dataset)} samples de computer use.")

        logger.info(f"--> Entrenando con {vl_epochs} épocas...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=vl_dataset,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_ratio=0.1,
                num_train_epochs=vl_epochs,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir="/tmp/outputs_vl",
                remove_unused_columns=False,        # CRÍTICO para VL
                dataset_text_field="",              # CRÍTICO — no hay campo de texto plano
                dataset_kwargs={"skip_prepare_dataset": True},  # CRÍTICO para VL
                dataloader_num_workers=0,           # OBLIGATORIO — PIL no serializable
                max_seq_length=2048,
            ),
        )

        trainer.train()
        del trainer
        trainer = None
        logger.info("✅ [Paso 3] Entrenamiento VL completado.")

        # ── PASO 4: Export GGUF + mmproj → MinIO → Webhook ──────────────────
        logger.info("--> [Paso 4] Exportando a GGUF + mmproj...")

        export_dir = f"/tmp/models/{tenant_id}-vl"
        import shutil
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)
        os.makedirs(export_dir, exist_ok=True)

        # Hilo heartbeat para monitorear el export (puede tardar 30-60 min)
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_export, args=(export_dir, stop_event), daemon=True
        )
        monitor_thread.start()

        try:
            os.environ["MAKEFLAGS"] = "-j1"
            os.environ["MAX_JOBS"] = "1"
            model.save_pretrained_gguf(
                export_dir,
                tokenizer,
                quantization_method="q4_k_m",
                maximum_memory_usage=0.5,
            )
        finally:
            stop_event.set()
            logger.info("--> Export finalizado.")

        # Unsloth agrega "_gguf" al nombre del directorio de salida real
        actual_output_dir = f"{export_dir}_gguf"

        output_gguf_s3 = None
        output_mmproj_s3 = None

        logger.info(f"--> Subiendo artefactos VL a MinIO bucket '{bucket_models}'...")
        for filename in os.listdir(actual_output_dir):
            filepath = os.path.join(actual_output_dir, filename)

            if filename.endswith(".gguf") and "mmproj" not in filename.lower():
                # Modelo principal con LoRA fusionado
                gguf_s3_name = f"{tenant_id}-vl.gguf"
                storage.upload_file(bucket_models, gguf_s3_name, filepath)
                output_gguf_s3 = f"{bucket_models}/{gguf_s3_name}"
                logger.info(f"    ✅ Modelo principal subido: {gguf_s3_name}")

            elif "mmproj" in filename.lower() and filename.endswith(".gguf"):
                # Vision projector (encoder visual)
                mmproj_s3_name = f"{tenant_id}-vl-mmproj.gguf"
                storage.upload_file(bucket_models, mmproj_s3_name, filepath)
                output_mmproj_s3 = f"{bucket_models}/{mmproj_s3_name}"
                logger.info(f"    ✅ Vision projector subido: {mmproj_s3_name}")

            elif "Modelfile" in filename:
                modelfile_s3_name = f"{tenant_id}-vl.Modelfile"
                storage.upload_file(bucket_models, modelfile_s3_name, filepath)

        if not output_gguf_s3:
            raise FileNotFoundError(f"No se encontró el GGUF principal en {actual_output_dir}")
        if not output_mmproj_s3:
            logger.warning(
                "⚠️  mmproj.gguf no encontrado — el computer use agent no tendrá capacidad visual."
            )

        # Limpiar archivo local temporal
        try:
            if os.path.exists(local_vl_path):
                os.remove(local_vl_path)
        except OSError:
            pass

        # Webhook OTA al Edge — con model_type="vision"
        if webhook_url:
            logger.info(f"--> 🔔 Disparando webhook OTA al Edge: {webhook_url}")
            try:
                payload = {
                    "model_tag": f"{tenant_id}-vl",
                    "model_type": "vision",
                    "mmproj_tag": f"{tenant_id}-vl-mmproj" if output_mmproj_s3 else None,
                }
                requests.post(
                    webhook_url,
                    json=payload,
                    headers={"x-api-key": settings.API_KEY},
                    timeout=10,
                )
            except Exception as e:
                logger.error(f"[VL-MLOps] Webhook OTA falló: {e}")

        logger.info(f"✅ [VL-MLOps] PIPELINE VL COMPLETADO para {tenant_id}.")
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "model_gguf_s3": output_gguf_s3,
            "mmproj_gguf_s3": output_mmproj_s3,
        }

    except Exception as e:
        logger.error(f"[VL-MLOps] ERROR EN PIPELINE VL: {str(e)}")
        return {"status": "error", "detail": str(e)}

    finally:
        # Limpieza VRAM — siempre
        logger.info("--> Ejecutando limpieza de VRAM...")
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("--> VRAM liberada.")


# ── Funciones auxiliares ─────────────────────────────────────────────────────

def _aggregate_vl_datasets(storage, bucket: str, tenant_id: str, output_path: str) -> bool:
    """
    Descarga y concatena todos los archivos JSONL del bucket datalake-vl
    que coincidan con el prefix del tenant_id.
    Retorna True si encontró al menos un archivo con datos.
    """
    if os.path.exists(output_path):
        os.remove(output_path)

    prefix = f"{tenant_id}_vl_"
    objects = storage.list_objects(bucket, prefix=prefix)
    found = False

    with open(output_path, "ab") as master_file:
        for obj in objects:
            temp_path = f"/tmp/{obj.object_name}"
            if storage.download_file(bucket, obj.object_name, temp_path):
                with open(temp_path, "rb") as f:
                    chunk = f.read()
                    if chunk.strip():
                        master_file.write(chunk)
                        master_file.write(b"\n")
                        found = True
                        logger.info(f"    - Incluido: {obj.object_name}")
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    return found


def _load_vl_dataset(jsonl_path: str) -> "HFDataset":
    """
    Carga el JSONL de datos VL y lo convierte a formato HuggingFace Dataset.

    Formato esperado de cada entry en el JSONL:
    {
      "messages": [
        {"role": "user", "content": [
          {"type": "image"},
          {"type": "text", "text": "Instrucción de alto nivel del Orchestrator"}
        ]},
        {"role": "assistant", "content": [
          {"type": "text", "text": "{\"type\":\"click\",\"x\":1450,\"y\":280}"}
        ]}
      ],
      "images": ["<base64_png_string>"]
    }

    Returns:
        HuggingFace Dataset con columnas: messages (list), images (list of PIL.Image)
    """
    from datasets import Dataset as HFDataset
    from PIL import Image as PILImage
    import io
    import base64

    raw_entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"[VL Dataset] JSON inválido ignorado: {line[:80]}...")

    logger.info(f"    Raw entries leídos: {len(raw_entries)}")

    processed = []
    skipped = 0
    for entry in raw_entries:
        try:
            images_b64 = entry.get("images", [])
            if not images_b64:
                skipped += 1
                continue

            pil_images = []
            for b64_str in images_b64:
                if b64_str:
                    img_bytes = base64.b64decode(b64_str)
                    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_images.append(img)

            if not pil_images:
                skipped += 1
                continue

            processed.append({
                "messages": entry["messages"],
                "images": pil_images,
            })

        except Exception as e:
            logger.warning(f"[VL Dataset] Entry ignorada por error: {e}")
            skipped += 1

    logger.info(f"    Procesados: {len(processed)} samples válidos | Ignorados: {skipped}")

    if not processed:
        raise ValueError(
            "El dataset VL no contiene samples válidos (con imágenes decodificables). "
            "Verifica el formato del JSONL y que las imágenes estén en base64 correcto."
        )

    return HFDataset.from_list(processed)


def _monitor_export(directory: str, stop_event: threading.Event):
    """Hilo heartbeat para monitorear el progreso del export GGUF (cada 30s)."""
    while not stop_event.is_set():
        total_bytes = 0
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for fname in files:
                    try:
                        total_bytes += os.path.getsize(os.path.join(root, fname))
                    except OSError:
                        pass
        total_mb = total_bytes / (1024 * 1024)
        logger.info(f"--> [VL Export] Convergiendo GGUF: {total_mb:.1f} MB escritos...")
        time.sleep(30)
