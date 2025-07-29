from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from dotenv import load_dotenv
load_dotenv()

# Define the models and their configurations

MODELS = {
    "gemma": {
        "hf_name": os.getenv("GEMMA_MODEL_NAME"),
        "local_path": os.getenv("MODELS_BASE_DIR") + os.getenv("GEMMA_BASE_MODEL_DIR")
    },
    "llama3": {
        "hf_name": os.getenv("LLAMA_MODEL_NAME"),
        "local_path": os.getenv("MODELS_BASE_DIR") + os.getenv("LLAMA_BASE_MODEL_DIR")
    },
    "zephyr": {
        "hf_name": os.getenv("ZEPHYR_MODEL_NAME"),
        "local_path": os.getenv("MODELS_BASE_DIR") + os.getenv("ZEPHYR_BASE_MODEL_DIR")
    },
}

for model_key, model_info in MODELS.items():
    print(f"Descargando modelo y tokenizer base para: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"])
    tokenizer.save_pretrained(model_info["local_path"])

    model = AutoModelForCausalLM.from_pretrained(model_info["hf_name"])
    model.save_pretrained(model_info["local_path"])
    print(f"Guardado en: {model_info['local_path']}\n")
