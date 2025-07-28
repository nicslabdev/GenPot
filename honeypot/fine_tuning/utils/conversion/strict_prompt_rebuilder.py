# strict_prompt_rebuilder.py
# Reescribe los prompts de evaluación para exigir respuestas estrictamente en formato JSON
# Utiliza las mismas columnas que 'evaluate_finetuned_models.csv' y genera prompts nuevos

import pandas as pd
import json

# Leer las peticiones originales (usadas en evaluación)
df = pd.read_csv("results/eval_finetuned_models.csv")

# Obtener combinaciones únicas de API + method + params
unique_prompts = df[["api", "method", "params"]].drop_duplicates()

# Regenerar prompts más estrictos
def build_prompt(row):
    return (
        "You are a Synology NAS running DSM 6, simulating a RESTful API response.\n"
        "You must respond with a valid JSON object only. No introduction, no explanation, no formatting notes.\n"
        "Start directly with an opening curly brace. Do not include any markdown or description.\n\n"
        f"API: {row['api']}\nMethod: {row['method']}\nParams: {row['params']}"
    )

unique_prompts["strict_prompt"] = unique_prompts.apply(build_prompt, axis=1)

# Exportar como referencia
unique_prompts.to_csv("results/strict_prompts.csv", index=False)

print("\n✅ Prompts estrictos generados y guardados en 'results/strict_prompts.csv'")
