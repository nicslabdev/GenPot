# evaluate_finetuned_models.py
# This script evaluates multiple fine-tuned models using a fixed set of realistic prompts based on the Synology NAS API.
# For each model, it generates responses, calculates if they are valid JSON, measures their length and response time.
# Exports results in JSONL and CSV format for further analysis.

import torch
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd

# Fine-tuned models to evaluate (updated)
model_paths = {
    "gemma": "./models/gemma_finetuned",
    "deepseek": "./models/deepseek_finetuned",
    "yi": "./models/yi_finetuned",
    "starcoder2": "./models/starcoder2_finetuned"
}

# Common evaluation prompts (realistic business environment)
prompts = [
    "API: SYNO.FileStation.List\nMethod: list_share\nParams: {}\nResponse:",
    "API: SYNO.FileStation.List\nMethod: list\nParams: {\"folder_path\": \"/volume1/Finance/Reports\"}\nResponse:",
    "API: SYNO.FileStation.Create\nMethod: create\nParams: {\"folder_path\": \"/volume1/Marketing/NewCampaign\"}\nResponse:",
    "API: SYNO.FileStation.Delete\nMethod: delete\nParams: {\"path\": \"/volume1/HR/Confidential/employee_list.pdf\"}\nResponse:",
    "API: SYNO.FileStation.Rename\nMethod: rename\nParams: {\"path\": \"/volume1/Projects/ProjectA/plan.txt\", \"name\": \"final_plan.txt\"}\nResponse:",
    "API: SYNO.FileStation.Move\nMethod: move\nParams: {\"from\": \"/volume1/Temp/invoice.tmp\", \"to\": \"/volume1/Accounting/Invoices/2023/\"}\nResponse:",
    "API: SYNO.FileStation.Copy\nMethod: copy\nParams: {\"from\": \"/volume1/Docs/security_policy.pdf\", \"to\": \"/volume1/Compliance/\"}\nResponse:",
    "API: SYNO.FileStation.Search\nMethod: start\nParams: {\"folder_path\": \"/volume1/Legal/Contracts\", \"pattern\": \"*.docx\"}\nResponse:",
    "API: SYNO.Core.System\nMethod: get\nParams: {}\nResponse:",
    "API: SYNO.Core.SyslogClient\nMethod: list\nParams: {}\nResponse:"
]

# Evaluation and storage of results
results = []

for name, path in model_paths.items():
    print(f"\n🧠 Evaluating model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}...")
        start_time = time.time()
        output = generate(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']
        duration = time.time() - start_time

        is_valid_json = True
        try:
            json.loads(output)
        except json.JSONDecodeError:
            is_valid_json = False

        results.append({
            "model": name,
            "prompt": prompt,
            "response": output,
            "length": len(output),
            "is_valid_json": is_valid_json,
            "response_time": duration,
            "timestamp": time.time()
        })

# Save results
Path("results").mkdir(exist_ok=True)
with open("results/eval_finetuned_models.jsonl", "w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")

pd.DataFrame(results).to_csv("results/eval_finetuned_models.csv", index=False)
print("\n✅ Results saved in 'results/eval_finetuned_models.csv'")
