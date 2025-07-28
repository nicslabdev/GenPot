from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "gemma": {
        "hf_name": "google/gemma-7b-it",
        "local_path": "/mnt/AI-DATA/jzamoraru_tfm/models/gemma_base"
    },
    "llama3": {
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "local_path": "/mnt/AI-DATA/jzamoraru_tfm/models/llama3_base"
    },
    "zephyr": {
        "hf_name": "HuggingFaceH4/zephyr-7b-beta",
        "local_path": "/mnt/AI-DATA/jzamoraru_tfm/models/zephyr_base"
    },
}

for model_key, model_info in MODELS.items():
    print(f"Descargando modelo y tokenizer base para: {model_key}")
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"])
    tokenizer.save_pretrained(model_info["local_path"])

    model = AutoModelForCausalLM.from_pretrained(model_info["hf_name"])
    model.save_pretrained(model_info["local_path"])
    print(f"Guardado en: {model_info['local_path']}\n")
