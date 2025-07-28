#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# GPU stats
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
    )
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info_pre = nvmlDeviceGetMemoryInfo(handle)
    mem_pre = info_pre.used
    pynvml_available = True
except Exception as e:
    print(f"pynvml no disponible ({e}), no mediremos VRAM ni potencia.")
    pynvml_available = False
    mem_pre = None

# Configuración
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "/mnt/AI-DATA/jzamoraru_tfm/models/llama3_finetuned"
DATASET_PATH = "llama3_synology_dataset.txt"

# Cuantización 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Carga modelo y LoRA
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
base_model = prepare_model_for_kbit_training(base_model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_config)

# Cargar dataset
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().split("<|begin_of_text|>")
    examples = [r.strip() for r in raw if r.strip()]
    return {"text": examples}

data_dict = load_dataset(DATASET_PATH)
dataset = Dataset.from_dict(data_dict)

# Tokenización
def tokenize_fn(ex):
    tokens = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=2048
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# Split train / validation
splits = tokenized.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds  = splits["test"]

# Entrenamiento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to="none",
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=default_data_collator
)

# Monitorización de potencia
power_samples = []
stop_flag = False
def monitor_power():
    while not stop_flag:
        try:
            power = nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
            power_samples.append(power)
        except:
            pass
        time.sleep(2)

if pynvml_available:
    thread = threading.Thread(target=monitor_power)
    thread.start()

# Entrena
start = time.time()
trainer.train()
end = time.time()
training_time = end - start

# Detiene monitor
if pynvml_available:
    stop_flag = True
    thread.join()
    avg_power = sum(power_samples) / len(power_samples) if power_samples else None
    energy_wh = avg_power * (training_time / 3600) if avg_power else None
else:
    avg_power = None
    energy_wh = None

# VRAM post
if pynvml_available:
    info_post = nvmlDeviceGetMemoryInfo(handle)
    vram_used_mb = (info_post.used - mem_pre) / 1024**2
else:
    vram_used_mb = None

# Loss final
final_loss = None
for entry in reversed(trainer.state.log_history):
    if "loss" in entry:
        final_loss = entry["loss"]
        break

# Carbon footprint
factor = 150  # gCO2/kWh
if energy_wh:
    carbon_kg = (energy_wh/1000) * factor / 1000
else:
    carbon_kg = None

# Guarda stats
stats = {
    "modelo": MODEL_NAME,
    "output_dir": OUTPUT_DIR,
    "dataset_samples": len(dataset),
    "epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "max_length": 2048,
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
    "quantization": "4bit-nf4",
    "training_time_sec": round(training_time, 2),
    "vram_used_mb": round(vram_used_mb, 2) if vram_used_mb else None,
    "final_loss": final_loss,
    "energy_wh": round(energy_wh, 2) if energy_wh else None,
    "co2_kg": round(carbon_kg, 4) if carbon_kg else None
}

with open(os.path.join(OUTPUT_DIR, "training_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "training_log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print("Entrenamiento Llama3 completado. Stats:")
print(json.dumps(stats, indent=2))
