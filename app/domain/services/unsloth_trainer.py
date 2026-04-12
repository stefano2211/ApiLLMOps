import gc
import json
import logging
import os
import shutil
import tarfile
from datetime import datetime

import requests
from celery import shared_task
from loguru import logger

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
        # CRÍTICO: Usar SFTConfig en vez de TrainingArguments.
        # SFTConfig hereda de TrainingArguments pero gestiona dataset_text_field
        # y la tokenización internamente, evitando que el procesador multimodal
        # de Qwen3.5 interprete texto como imágenes.
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        logger.error(f"[MLOps] LIBRERÍAS GPU FATAL ERROR: {e}. Se cancela el Job.")
        raise RuntimeError(f"Missing GPU libraries: {e}") from e

    from app.persistence.storage import storage
    
    bucket_datalake = settings.S3_BUCKET_DATALAKE
    object_name = f"{tenant_id}_master.jsonl"
    local_dataset_path = f"/tmp/{object_name}"
    
    model = None
    tokenizer = None
    trainer = None
    export_dir = None
    tar_path = None
    
    try:
        # --- PASO 1: Chequeo de Data y Carga del Modelo Base ---
        self.update_state(state="PROGRESS", meta={"step": 1, "msg": f"Agregando datasets de {tenant_id}"})
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
            
        self.update_state(state="PROGRESS", meta={"step": 1, "msg": f"Cargando modelo base {base_model}"})
        logger.info(f"--> [Paso 1.1] Cargando modelo base {base_model} en 16-bit LoRA...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=1024, # Reducido a 1024 para evitar OOM
            dtype=None, # Auto-detecta fp16 o bf16
            load_in_4bit=True, # QLoRA (4-bit): ~75% menos VRAM vs LoRA 16-bit, marginalmente menos preciso.
                           # Unsloth recomienda load_in_4bit=False (LoRA puro) para Qwen3.5 si hay VRAM suficiente.
        )

        # --- PASO 2: Formateo de Datos ---
        self.update_state(state="PROGRESS", meta={"step": 2, "msg": "Formateando dataset a ChatML"})
        logger.info(f"--> [Paso 2] Cargando Dataset {local_dataset_path} a ChatML...")
        dataset = load_dataset("json", data_files={"train": local_dataset_path}, split="train")

        logger.info(f"--> [Paso 2.1] Columnas crudas detectadas: {dataset.column_names}")

        # ---------------------------------------------------------------
        # Normalizar formato mixto ANTES de standardize_sharegpt.
        #
        # El pipeline recibe datos de MÚLTIPLES fuentes con formatos distintos:
        #   1. DbCollector (formatter.py) → ShareGPT:
        #      {"conversations": [{"from":"user","value":"..."},{"from":"assistant","value":"..."}]}
        #   2. e2e_test_runner / Edge mock → OpenAI ChatML:
        #      {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
        #
        # standardize_sharegpt() REQUIERE columna "conversations" con "from"/"value".
        # Aquí unificamos todo a ese formato ANTES de llamarla.
        # ---------------------------------------------------------------
        def _normalize_to_sharegpt(example):
            convo = example.get("conversations")
            msgs = example.get("messages")

            # Caso 1: Ya tiene "conversations" con "from"/"value" → OK
            if convo and isinstance(convo, list) and len(convo) > 0:
                first = convo[0]
                if isinstance(first, dict) and "from" in first:
                    return {"conversations": convo}
                # Edge case: conversations con role/content
                if isinstance(first, dict) and "role" in first:
                    return {"conversations": [
                        {"from": m.get("role", "user"), "value": m.get("content", "")}
                        for m in convo if m is not None
                    ]}

            # Caso 2: Tiene "messages" (OpenAI ChatML) → Convertir a ShareGPT
            if msgs and isinstance(msgs, list) and len(msgs) > 0:
                return {"conversations": [
                    {"from": m.get("role", "user"), "value": m.get("content", "")}
                    for m in msgs if m is not None
                ]}

            # Fila inválida
            return {"conversations": []}

        dataset = dataset.map(_normalize_to_sharegpt)
        dataset = dataset.filter(lambda x: len(x["conversations"]) > 0)
        # Limpiar columnas sobrantes (messages, etc.) ANTES de standardize
        cols_to_drop = [c for c in dataset.column_names if c != "conversations"]
        if cols_to_drop:
            dataset = dataset.remove_columns(cols_to_drop)
        logger.info(f"--> [Paso 2.2] Dataset normalizado: {len(dataset)} filas. Columnas: {dataset.column_names}")

        # standardize_sharegpt: convierte from/value → role/content para apply_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        dataset = standardize_sharegpt(dataset)

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in convos
            ]
            return {"text": texts}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        # CRÍTICO: Tokenización manual con argumentos NOMBRADOS (text=...)
        # El tokenizer de Qwen3.5 es multimodal. Si le pasamos el texto posicionalmente
        # (ej. tokenizer(examples["text"])), lo interpreta erróneamente como una imagen
        # y falla con "Incorrect image source". Esto también arregla el error de TRL
        # donde falla el tokenizado interno al usar dataset_text_field.
        def tokenize_func(examples):
            return tokenizer(
                text=examples["text"], 
                truncation=True, 
                max_length=1024, 
                padding=False
            )
            
        dataset = dataset.map(tokenize_func, batched=True)
        logger.info(f"--> [Paso 2.3] Dataset tokenizado manualmente. Columnas: {dataset.column_names}")

        # Limpiar columna "text" para que solo queden input_ids y attention_mask
        if "text" in dataset.column_names:
            dataset = dataset.remove_columns(["text"])
        logger.info(f"--> [Paso 2.4] Dataset final para Trainer. Columnas: {dataset.column_names}")

        # --- PASO 3: Inyección LoRA ---
        self.update_state(state="PROGRESS", meta={"step": 3, "msg": "Inyectando adaptadores LoRA"})
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
        # Siguiendo receta oficial de Unsloth para Qwen3.5:
        # https://unsloth.ai/docs/models/qwen3.5/fine-tune#quickstart
        self.update_state(state="PROGRESS", meta={"step": 4, "msg": f"Entrenando modelo — {epochs} épocas"})
        logger.info(f"--> [Paso 4] Entrenando modelo (Settings SOTA). Epochs: {epochs}...")
        
        # CRÍTICO: Usar SFTConfig — NO TrainingArguments.
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                # SIN dataset_text_field, ya tokenizado
                max_seq_length=1024,
                dataset_num_proc=1,              # Qwen3.5 oficial usa 1
                packing=False,
                per_device_train_batch_size=1, 
                gradient_accumulation_steps=4,
                warmup_steps=10,
                num_train_epochs=epochs,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",              # Receta oficial Qwen3.5
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir="/tmp/outputs",
            ),
        )

        logger.info("--> Iniciando Pila Tensorial (Trainer)...")
        trainer_stats = trainer.train()

        # --- PASO 5: Exportación LoRA y Compresión (CRÍTICO PARA vLLM) ---
        self.update_state(state="PROGRESS", meta={"step": 5, "msg": "Exportando LoRA safetensors y subiendo a S3"})
        logger.info("--> [Paso 5] Guardando LoRA en safetensors...")
        export_dir = f"/tmp/models/{tenant_id}-v2-lora"
        
        # Limpiar residuos fantasmas
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)
            
        os.makedirs(export_dir, exist_ok=True)
        
        logger.info(f"--> Generando Safetensors (LoRA): {export_dir}")
        
        # Guardado instántaneo de adaptadores LoRA (~100MB)
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        
        # Comprimir en tar.gz para subida a S3 y fácil descarga OTA
        tar_path = f"{export_dir}.tar.gz"
        logger.info(f"--> Comprimiendo {export_dir} en {tar_path}...")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(export_dir, arcname=f"{tenant_id}-v2")
        
        # Subir artefacto zipeado a S3
        bucket_models = settings.S3_BUCKET_MODELS
        output_tar_s3 = None
        
        logger.info("--> Subiendo Tarball a S3 / Minio...")
        tar_s3_name = f"{tenant_id}-v2-lora.tar.gz"
        storage.upload_file(bucket_models, tar_s3_name, tar_path)
        output_tar_s3 = f"{bucket_models}/{tar_s3_name}"

        # Limpieza TMP
        try:
            os.remove(local_dataset_path)
            os.remove(tar_path)
            shutil.rmtree(export_dir, ignore_errors=True)
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
                    json={"model_tag": f"{tenant_id}-v2", "model_type": "text"},
                    headers=webhook_headers,
                    timeout=10,
                )
            except Exception as e:
                logger.error(f"Webhook failed: {e}")
        
        logger.info(f"✅ [MLOps] PIPELINE COMPLETADO EXITOSAMENTE para {tenant_id}. Adaptador LoRA .tar.gz disponible en S3.")
                
        return {
            "status": "success", 
            "tenant_id": tenant_id, 
            "output_s3": output_tar_s3
        }

    except Exception as e:
        logger.error(f"[MLOps] ERROR EN PIPELINE DE ENTRENAMIENTO: {str(e)}")
        raise  # Let Celery mark the task as FAILURE so status endpoint reflects the real state

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
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        # Limpieza de archivos temporales (también en caso de fallo)
        for _path in [local_dataset_path]:
            try:
                if os.path.exists(_path):
                    os.remove(_path)
            except OSError:
                pass
        if export_dir and os.path.exists(export_dir):
            shutil.rmtree(export_dir, ignore_errors=True)
        if tar_path and os.path.exists(tar_path):
            try:
                os.remove(tar_path)
            except OSError:
                pass

        logger.info("--> GPU Memory y archivos temporales liberados.")
