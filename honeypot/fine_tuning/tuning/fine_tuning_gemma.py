#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import threading
from dotenv import load_dotenv
load_dotenv()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

exit()

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

# ==== GPU stats via NVML ====
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
    print(f"pynvml not available ({e}), VRAM and power will not be measured.")
    pynvml_available = False
    mem_pre = None

# ==== Configuration ====
MODEL_NAME   = os.getenv("GEMMA_MODEL_NAME")
OUTPUT_DIR   = os.getenv("MODELS_BASE_DIR") + "/" + os.getenv("GEMMA_FINETUNED_DIR_NAME")
DATASET_PATH = os.getenv("DATASETS_BASE_DIR") + "/" + os.getenv("GEMMA_DATASET_NAME")

# ==== 4-bit quantization ====
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ==== Load model + LoRA ====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
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

# ==== Load and format dataset ====
def load_dataset(path):
    """Split by turns <start_of_turn>…<end_of_turn>."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Each example is "user" + "model" turns
    chunks = text.split("<start_of_turn>user")[1:]
    examples = []
    for chunk in chunks:
        # retrieve user-turn until <end_of_turn>
        user_part, rest = chunk.split("<end_of_turn>", 1)
        model_part = rest.split("<start_of_turn>model",1)[1].split("<end_of_turn>",1)[0]
        prompt = user_part.strip()
        response = model_part.strip()
        # concatenate for the model
        examples.append(f"<|prompt|>{prompt}\n<|response|>{response}")
    return {"text": examples}

data_dict = load_dataset(DATASET_PATH)
dataset = Dataset.from_dict(data_dict)

# ==== Tokenization ====
def tokenize_fn(ex):
    tokens = tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# ==== Train/validation split ====
splits = tokenized.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds  = splits["test"]

# ==== Training arguments ====
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

# ==== Power monitor ====
power_samples = []
stop_flag = False
def monitor_power():
    while not stop_flag:
        try:
            p = nvmlDeviceGetPowerUsage(handle) / 1000  # mW→W
            power_samples.append(p)
        except:
            pass
        time.sleep(2)

if pynvml_available:
    thread = threading.Thread(target=monitor_power)
    thread.start()

# ==== Training ====  
start = time.time()
trainer.train()
end = time.time()
training_time = end - start

# ==== Stop monitor ====  
if pynvml_available:
    stop_flag = True
    thread.join()
    avg_power = sum(power_samples)/len(power_samples) if power_samples else None
    energy_wh = avg_power * (training_time/3600) if avg_power else None
else:
    avg_power = None
    energy_wh = None

# ==== VRAM post-training ====  
if pynvml_available:
    info_post = nvmlDeviceGetMemoryInfo(handle)
    vram_used_mb = (info_post.used - mem_pre)/1024**2
else:
    vram_used_mb = None

# ==== Final loss ====  
final_loss = None
for entry in reversed(trainer.state.log_history):
    if "loss" in entry:
        final_loss = entry["loss"]
        break

# ==== Carbon footprint ====  
factor = 150  # gCO₂/kWh
if energy_wh:
    carbon_kg = (energy_wh/1000)*factor/1000
else:
    carbon_kg = None

# ==== Save statistics ====  
stats = {
    "model": MODEL_NAME,
    "output_dir": OUTPUT_DIR,
    "dataset_samples": len(dataset),
    "epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "max_length": 512,
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
    "quantization": "4bit-nf4",
    "training_time_sec": round(training_time, 2),
    "vram_used_mb": round(vram_used_mb, 2) if vram_used_mb else None,
    "final_loss": final_loss,
    "energy_wh": round(energy_wh, 2) if energy_wh else None,
    "co2_kg": round(carbon_kg, 4) if carbon_kg else None
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "training_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "training_log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print("=== Gemma fine-tuning completed ===")
print(json.dumps(stats, indent=2))
