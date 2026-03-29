from celery import shared_task
from loguru import logger
import time
import requests
import os
import gc
import json
import logging
import threading
from datetime import datetime
from app.core.config import settings

@shared_task(bind=True)
def start_finetuning_task(self, tenant_id: str, base_model: str, epochs: int, webhook_url: str):
    """
    Script de entrenamiento asincrónico (Celery Worker).
    Pipeline de 5 pasos basado en Unsloth + llama.cpp para entorno Edge.
    """
    # Habilitar descarga acelerada desde HuggingFace Hub al inicio
    # Aplica tanto al from_pretrained (Paso 1) como al export GGUF (Paso 5)
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    logger.info(f"[MLOps] Starting Production Pipeline for {tenant_id} on {base_model}")
    
    # Intenta cargar librerías GPU, si fallan atrapa la excepción para no crashear celery
    try:
        import torch
        # Hotfix para PyTorch 2.4.0 + unsloth_zoo: Forzar la carga de los submodulos del compilador
        import torch._dynamo
        try:
            import torch._inductor.config
        except Exception:
            pass

        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt
        from datasets import load_dataset
        from transformers import TrainingArguments
        # DataCollatorForCompletionOnlyLM es opcional — cambia de ubicación según versión de TRL
        _DataCollator = None
        try:
            from trl import SFTTrainer, DataCollatorForCompletionOnlyLM as _DataCollator
        except ImportError:
            try:
                from trl import SFTTrainer
                from trl.trainer.utils import DataCollatorForCompletionOnlyLM as _DataCollator
            except ImportError:
                from trl import SFTTrainer
                logger.warning("[MLOps] DataCollatorForCompletionOnlyLM no disponible. Entrenando sobre texto completo (fallback).")
    except ImportError as e:
        logger.error(f"[MLOps] LIBRERÍAS GPU FATAL ERROR: {e}. Se cancela el Job.")
        return {"status": "error", "detail": "Missing GPU libraries or Torch"}

    from app.persistence.storage import storage
    
    bucket_datalake = settings.S3_BUCKET_DATALAKE
    object_name = f"{tenant_id}_master.jsonl"
    local_dataset_path = f"/tmp/{object_name}"
    
    model = None
    tokenizer = None
    trainer = None
    
    try:
        # --- PASO 1: Chequeo de Data y Carga del Modelo Base ---
        logger.info(f"--> [Paso 1] Agregando todos los datasets de {tenant_id}...")
        
        objects = storage.list_objects(bucket_datalake, prefix=f"{tenant_id}_")
        found_data = False
        
        # Limpiar local si existe de corridas previas
        if os.path.exists(local_dataset_path):
            os.remove(local_dataset_path)

        with open(local_dataset_path, "ab") as master_file:
            for obj in objects:
                temp_tool_file = f"/tmp/{obj.object_name}"
                if storage.download_file(bucket_datalake, obj.object_name, temp_tool_file):
                    with open(temp_tool_file, "rb") as f:
                        master_file.write(f.read())
                        master_file.write(b"\n") # Asegurar salto entre herramientas
                    os.remove(temp_tool_file)
                    found_data = True
                    logger.info(f"    - Incluido: {obj.object_name}")

        if not found_data:
            raise FileNotFoundError(f"No se encontraron datasets para {tenant_id} en S3.")
            
        logger.info(f"--> [Paso 1.1] Cargando modelo base {base_model} en 4-bits...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=1024, # Safe bound para secuencias de 500-600 tokens
            dtype=None, # Auto-detecta fp16 o bf16
            load_in_4bit=True,
        )

        # --- PASO 2: Formateo de Datos ---
        logger.info(f"--> [Paso 2] Cargando Dataset {local_dataset_path} a ChatML...")
        dataset = load_dataset("json", data_files={"train": local_dataset_path}, split="train")

        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}

        dataset = standardize_sharegpt(dataset)
        dataset = dataset.map(formatting_prompts_func, batched=True)

        # --- PASO 3: Inyección LoRA ---
        logger.info("--> [Paso 3] Inyectando LoRA (QLoRA-All, r=16, alpha=32)...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth", # Usa un 30% menos de VRAM
            random_state=3407,
            use_rslora=False,
        )

        # --- PASO 4: Bucle de Entrenamiento (Data Replay) ---
        logger.info(f"--> [Paso 4] Entrenando modelo (Settings SOTA). Epochs: {epochs}...")
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=1024, # Safe bound para evitar el crash de Dynamo con secuencias > 512
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1, 
                gradient_accumulation_steps=8, # Aumentado de 4 a 8
                warmup_ratio=0.1,
                num_train_epochs=epochs,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir="/tmp/outputs",
            ),
        )

        # Entrenar solo en las completaciones multi-turno (SOTA)
        if _DataCollator is not None:
            response_template = "<|im_start|>assistant\n"
            collator = _DataCollator(response_template, tokenizer=tokenizer)
            trainer.data_collator = collator
            logger.info("---> Usando DataCollator de completaciones (SOTA multi-turno).")
        else:
            logger.warning("---> DataCollator no disponible. Entrenando sobre texto completo.")

        logger.info("--> Iniciando Pila Tensorial (Trainer)...")
        trainer_stats = trainer.train()

        # --- PASO 5: Fusión y Exportación (CRÍTICO PARA OLLAMA) ---
        logger.info("--> [Paso 5] Fusión LoRA y conversión nativa GGUF/Modelfile...")
        export_dir = f"/tmp/models/{tenant_id}-v2"
        
        # Limpiar residuos fantasmas (stale) de runs de entrenamiento previos
        import shutil
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)
            
        os.makedirs(export_dir, exist_ok=True)
        
        logger.info(f"--> Auto-Generando GGUF nativo en: {export_dir}")
        
        # Hilo de monitoreo para visibilidad de progreso (Heartbeat avanzada)
        def monitor_export_progress(directory, stop_event):
            cache_dir = "/app/data/huggingface_cache/hub"
            while not stop_event.is_set():
                total_export_size = 0
                if os.path.exists(directory):
                    for root, dirs, files in os.walk(directory):
                        for f in files:
                            try:
                                total_export_size += os.path.getsize(os.path.join(root, f))
                            except OSError:
                                pass
                
                total_cache_size = 0
                if os.path.exists(cache_dir):
                    for root, dirs, files in os.walk(cache_dir):
                        for f in files:
                            try:
                                total_cache_size += os.path.getsize(os.path.join(root, f))
                            except OSError:
                                pass
                
                export_mb = total_export_size / (1024 * 1024)
                cache_gb = total_cache_size / (1024 * 1024 * 1024)
                
                if export_mb > 15: # Superado el umbral de config/tokenizer inicial
                    logger.info(f"--> [Export Progress] Fusionando Pesos y Escribiendo GGUF: {export_mb:.2f} MB")
                else:
                    logger.info(f"--> [Export Progress] Paso 5 en Marcha: Descargando Pesos Base a Cache: {cache_gb:.2f} GB (Directorio export: {export_mb:.2f} MB)")
                
                time.sleep(30) # Cada 30 segundos

        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_export_progress, args=(export_dir, stop_event))
        monitor_thread.daemon = True # Asegura que no bloquee el cierre del worker
        monitor_thread.start()

        try:
            # Force llama.cpp compilation to use only 1 thread to strictly prevent WSL/Docker OOM crashes
            os.environ["MAKEFLAGS"] = "-j1"
            os.environ["MAX_JOBS"] = "1"
            
            model.save_pretrained_gguf(
                export_dir, 
                tokenizer, 
                quantization_method="q4_k_m",
                maximum_memory_usage=0.5
            )
        finally:
            stop_event.set()
            # No bloqueamos esperando el join para no retrasar la subida a S3
            logger.info("--> Exportación finalizada. Cerrando monitoreo.")

        
        # Buscar el archivo .gguf generado y el Modelfile en el directorio
        bucket_models = settings.S3_BUCKET_MODELS
        output_gguf_s3 = None
        
        logger.info("--> Subiendo artefactos nativos a S3...")
        # Unsloth adds a "_gguf" suffix to the export directory magically.
        actual_output_dir = f"{export_dir}_gguf"
        
        for file in os.listdir(actual_output_dir):
            file_path = os.path.join(actual_output_dir, file)
            if file.endswith(".gguf"):
                gguf_s3_name = f"{tenant_id}-v2.gguf"
                storage.upload_file(bucket_models, gguf_s3_name, file_path)
                output_gguf_s3 = f"{bucket_models}/{gguf_s3_name}"
            elif "Modelfile" in file:
                modelfile_s3_name = f"{tenant_id}-v2.Modelfile"
                storage.upload_file(bucket_models, modelfile_s3_name, file_path)

        # Limpieza TMP
        try:
            os.remove(local_dataset_path)
            # clean up both directories if possible
        except OSError:
            pass

        # --- Cierre de Ciclo: OTA Update Webhook ---
        if webhook_url:
            logger.info(f"--> Lanzando OTA Webhook hacia Edge: {webhook_url}")
            try:
                # x-api-key header is REQUIRED by IndustrialBackend /mlops/webhook/model-ready
                # Without it, the Edge responds 422 and the OTA update never triggers.
                webhook_headers = {"x-api-key": settings.API_KEY}
                requests.post(
                    webhook_url,
                    json={"model_tag": f"{tenant_id}-v2"},
                    headers=webhook_headers,
                    timeout=10,
                )
            except Exception as e:
                logger.error(f"Webhook failed: {e}")
        
        logger.info(f"✅ [MLOps] PIPELINE COMPLETADO EXITOSAMENTE para {tenant_id}. Modelo GGUF disponible en S3.")
                
        return {
            "status": "success", 
            "tenant_id": tenant_id, 
            "output_s3": output_gguf_s3
        }

    except Exception as e:
        logger.error(f"[MLOps] ERROR EN PIPELINE DE ENTRENAMIENTO: {str(e)}")
        return {"status": "error", "detail": str(e)}

    finally:
        # Limpieza crítica de VRAM pase lo que pase
        logger.info("--> Ejecutando Limpieza de Memoria GPU (Try-Finally)...")
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
            
        gc.collect()
        if 'torch' in locals() or 'torch' in globals():
            torch.cuda.empty_cache()
            
        logger.info("--> GPU Memory Liberada correctamente.")
