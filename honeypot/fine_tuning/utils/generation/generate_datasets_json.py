import pandas as pd
import json
import csv

# Read correctly escaped CSV
df = pd.read_csv("api_responses.csv", quoting=csv.QUOTE_ALL, escapechar='\\')

datasets = []

for _, row in df.iterrows():
    api = row["api"]
    method = row["method"]

    # now json.loads works correctly
    params = json.loads(row["params"])
    response = json.loads(row["response"])

    input_text = (
        f"[INST] Return valid JSON only. "
        f"API: {api} Method: {method} Params: {json.dumps(params)} [/INST]"
    )

    output_text = json.dumps(response)

    datasets.append({
        "input": input_text,
        "output": output_text
    })

# Save to JSONL file
with open("dataset_zephyr.jsonl", "w") as f:
    for entry in datasets:
        f.write(json.dumps(entry) + "\n")

print("✅ Dataset correctly generated.")
