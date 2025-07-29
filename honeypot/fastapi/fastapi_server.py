# fastapi_models.py
# Servidor FastAPI que simula un endpoint de Synology NAS utilizando múltiples modelos LLM fine-tuneados (LoRA/PEFT)
#!/usr/bin/env python3

# Servidor FastAPI que simula un endpoint de Synology NAS
# utilizando múltiples LLM fine-tuneados (LoRA/PEFT) con sets de hiperparámetros por modelo.

import os, json, random
from string import Template
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI()

# 1. CONFIG: rutas a base + adapters
MODEL_CONFIG = {
    "gemma": {
        "base": "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/models/gemma_base",
        "adapter": "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/models/gemma_finetuned",
    },
    "llama3": {
        "base": "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/models/llama3_base",
        "adapter": "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/models/llama3_finetuned",
    },
    "zephyr": {
        "base": "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/models/zephyr_base",
        "adapter": "/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/models/zephyr_finetuned",
    },
}

# 2. DEVICE + QUANT CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 if device.type=="cuda" else torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 3. CARGA DE MODELOS A GPU
loaded_models = {}
for name, paths in MODEL_CONFIG.items():
    print(f"Loading model {name}…")
    tok = AutoTokenizer.from_pretrained(paths["base"], trust_remote_code=True, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        paths["base"],
        quantization_config=QUANT_CONFIG,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, paths["adapter"], local_files_only=True)
    model.to(device).eval()
    loaded_models[name] = {"model": model, "tokenizer": tok, "device": device}

# 4. UTILIDADES DE PROMPTS
def load_prompt_template(model_name: str) -> str:
    path = os.path.join("/mnt/AI-DATA/alara/CiberIA_O1_A3/honeypot/fine_tuning/prompts", f"{model_name}.txt")
    with open(path, encoding="utf-8") as f:
        print(f"Loaded prompt template for {model_name} from {path}")
        return f.read()

def build_prompt(template_str: str, api: str, method: str, params: dict) -> str:
    return Template(template_str).safe_substitute(
        api=api, method=method, params=json.dumps(params, ensure_ascii=False)
    )

# 5. EXTRACCIÓN DE BLOQUE JSON
def extract_valid_json_block(text: str):
    candidates = []
    i = 0
    while True:
        i = text.find("{", i)
        if i<0: break
        depth = 0
        for j in range(i, len(text)):
            if text[j]=="{": depth+=1
            elif text[j]=="}":
                depth-=1
                if depth==0:
                    c = text[i:j+1]
                    try:
                        o = json.loads(c)
                        if "success" in o and ("data" in o or "error" in o):
                            candidates.append(o)
                    except: pass
                    break
        i += 1
    return max(candidates, key=lambda o: len(json.dumps(o))) if candidates else None

# 6. RESPUESTAS DE ERROR GENÉRICO
def default_synology_error():
    choices = [
        {"code":1050,"message":"Invalid API or parameters"},
        {"code":1100,"message":"Unknown error occurred"},
        {"code":1001,"message":"Unknown API or method"},
        {"code":408, "message":"Request timeout"}
    ]
    return {"success": False, "error": random.choice(choices)}

# 7. GENERACIÓN RAW
def generate_response(model_obj, tokenizer, device, prompt: str,
                      temperature, top_p, repetition_penalty, max_new_tokens) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model_obj.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 8. TRY + SETS POR MODELO
def try_generate_valid_response(model_obj, tokenizer, device, prompt: str, model_name: str):
    param_sets = {
        "gemma": [
            {"temperature":0.5, "top_p":0.9,  "repetition_penalty":1.0, "max_new_tokens":128},
            {"temperature":0.7, "top_p":0.95, "repetition_penalty":1.0, "max_new_tokens":256},
        ],
        "llama3": [
            {"temperature":0.7, "top_p":0.95, "repetition_penalty":1.0, "max_new_tokens":256},
            {"temperature":1.0, "top_p":0.99, "repetition_penalty":0.9, "max_new_tokens":384},
        ],
        "zephyr": [
            {"temperature":0.7, "top_p":0.95, "repetition_penalty":1.0, "max_new_tokens":512},
            {"temperature":0.4, "top_p":0.85, "repetition_penalty":1.2, "max_new_tokens":768},
        ],
    }[model_name]

    for p in param_sets:
        raw = generate_response(model_obj, tokenizer, device, prompt,
                                p["temperature"], p["top_p"],
                                p["repetition_penalty"], p["max_new_tokens"])
        js = extract_valid_json_block(raw)
        if js:
            return js
    return None

# 9. ENDPOINT SYNOPSY
@app.get("/webapi/entry.cgi")
async def syno_api(
    api: str = Query(...),
    method: str = Query(...),
    version: int = Query(...),
    model_name: str = Query("zephyr"),
    request: Request = None
):
    if model_name not in loaded_models:
        return default_synology_error()

    qp = dict(request.query_params)
    params = {k:v for k,v in qp.items() if k not in ("api","method","model_name")}
    params["version"] = version

    tmpl = load_prompt_template(model_name)
    prompt = build_prompt(tmpl, api, method, params)
    print(f"Generated prompt for {model_name}:\n{prompt}\n")
    res = try_generate_valid_response(
        loaded_models[model_name]["model"],
        loaded_models[model_name]["tokenizer"],
        loaded_models[model_name]["device"],
        prompt,
        model_name
    )
    print(f"Generated response for {model_name}:\n{json.dumps(res, indent=4, ensure_ascii=False) if res else '❌ No valid JSON generated'}\n")
    return res or default_synology_error()

# 10. ENDPOINT AUX PARA PRUEBAS
class GenerationRequest(BaseModel):
    prompt: str
    model: str
    temperature: float=0.7
    top_p: float=0.95
    repetition_penalty: float=1.0
    max_new_tokens: int=256

@app.post("/generate")
async def gen_from_prompt(req: GenerationRequest):
    if req.model not in loaded_models:
        return default_synology_error()
    return try_generate_valid_response(
        loaded_models[req.model]["model"],
        loaded_models[req.model]["tokenizer"],
        loaded_models[req.model]["device"],
        req.prompt,
        req.model
    ) or default_synology_error()

@app.post("/generate_all")
async def gen_all(req: GenerationRequest):
    out = {}
    for m, d in loaded_models.items():
        out[m] = try_generate_valid_response(
            d["model"], d["tokenizer"], d["device"], req.prompt, m
        ) or default_synology_error()
    return out
