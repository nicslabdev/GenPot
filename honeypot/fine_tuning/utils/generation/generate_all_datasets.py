import pandas as pd
import json
import csv

# Load the extended CSV correctly
print("📥 Loading 'api_responses_extended.csv'...")
df = pd.read_csv("api_responses_extended.csv", quoting=csv.QUOTE_ALL, escapechar='\\')

# Model-specific prompt format
formats = {
    "zephyr": "[INST] Return valid JSON only. API: {api} Method: {method} Params: {params} [/INST]",
    "deepseek": "### Instruction:\nReturn valid JSON only.\nAPI: {api} Method: {method} Params: {params}\n\n### Response:\n",
    "starcoder2": "// Return valid JSON only\nAPI: {api} Method: {method} Params: {params}\nResponse:\n",
    "yi": "Human: Return valid JSON only.\nAPI: {api} Method: {method} Params: {params}\nAssistant:"
}

datasets = {model: [] for model in formats.keys()}

for _, row in df.iterrows():
    if pd.isna(row["params"]) or pd.isna(row["response"]):
        print(f"⚠️ Skipping incomplete row: {row.to_dict()}")
        continue

    api = row["api"]
    method = row["method"]
    params = json.loads(row["params"])
    response = json.loads(row["response"])

    for model, template in formats.items():
        input_text = template.format(api=api, method=method, params=json.dumps(params))
        output_text = json.dumps(response)

        datasets[model].append({
            "input": input_text,
            "output": output_text
        })

# Save datasets in JSONL format per model
for model, data in datasets.items():
    filename = f"dataset_{model}.jsonl"
    with open(filename, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Dataset generated: {filename}")

print("\n🎉 All datasets have been successfully generated for Zephyr, Deepseek, Starcoder2 and Yi.")
