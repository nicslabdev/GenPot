#This script analisis the answer from models and check the Levenstein distance between the expected result and the actual result answer from each model
#!/usr/bin/env python3
# script_analisis_respuestas_modelos.py

import os
import json
import glob
import pandas as pd
from Levenshtein import distance as lev

# ——— Configuración —————————————————————————————————
RESULTS_DIR = "results"
CSV_FILE    = os.path.join(RESULTS_DIR, "tiempos_respuesta.csv")
GOLD_FILE   = "gold.json"       # Puede ser dict mapping slug→respuesta o lista
MODELS      = ["gemma", "llama3", "zephyr"]
FIRST_MODEL = MODELS[0]

# ——— Carga de datos de tiempo ——————————————————————
df_times = pd.read_csv(CSV_FILE)

# ——— Extraer slugs en orden de petición para el primer modelo —————
df_first = df_times[df_times["modelo"] == FIRST_MODEL].sort_values("n_peticion")
slugs = df_first["slug_peticion"].tolist()

# ——— Carga del gold file ————————————————————————————
with open(GOLD_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, dict):
    # dict mapping slug->respuesta
    gold_map = raw.copy()
elif isinstance(raw, list):
    # lista de respuestas: emparejamos por posición con slugs[]
    gold_map = {
        slug: json.dumps(item, separators=(",", ":"))
        for slug, item in zip(slugs, raw)
    }
else:
    raise RuntimeError("gold.json debe ser dict o lista")

# ——— Evaluación ————————————————————————————————————
records = []
for model in MODELS:
    df_m = df_times[df_times["modelo"] == model].sort_values("n_peticion")
    for _, row in df_m.iterrows():
        slug = row["slug_peticion"]
        t    = row["tiempo_segundos"]

        # — cargar respuesta generada —
        pattern = os.path.join(RESULTS_DIR, model, f"*_{slug}.json")
        files = glob.glob(pattern)
        resp_text = ""
        if files:
            with open(files[0], "r", encoding="utf-8") as f:
                resp_text = f.read().strip()

        # — parseo JSON y flag de parseo correcto —
        try:
            resp_obj = json.loads(resp_text)
            ok_parse = True
        except Exception:
            resp_obj = {}
            ok_parse = False

        # — chequeo éxito/error coincide con gold —
        gold_obj = gold_map.get(slug, {})
        try:
            gold_parsed = json.loads(gold_obj) if isinstance(gold_obj, str) else gold_obj
        except:
            gold_parsed = gold_obj
        class_ok = ok_parse and (resp_obj.get("success") == gold_parsed.get("success"))

        # — Levenshtein sobre strings canónicas —
        a = json.dumps(resp_obj, sort_keys=True, separators=(",", ":"))
        b = json.dumps(gold_parsed, sort_keys=True, separators=(",", ":"))
        d_raw = lev(a, b)
        L = max(len(a), len(b)) or 1
        sim  = 1 - d_raw / L

        # — cobertura de campos —
        if ok_parse and isinstance(gold_parsed, dict):
            keys_r   = set(resp_obj.keys())
            keys_g   = set(gold_parsed.keys())
            key_match = len(keys_r & keys_g) / len(keys_g) if keys_g else None
        else:
            key_match = None

        records.append({
            "modelo":      model,
            "tiempo":      float(t),
            "ok_parse":    ok_parse,
            "class_ok":    class_ok,
            "dist_raw":    d_raw,
            "similitud":   sim,
            "key_match":   key_match,
        })

df_eval = pd.DataFrame(records)

# ——— Resumen por modelo ————————————————————————
summary = df_eval.groupby("modelo").agg({
    "tiempo":      "mean",
    "ok_parse":    "mean",
    "class_ok":    "mean",
    "similitud":   "mean",
    "key_match":   "mean",
}).rename(columns={
    "tiempo":    "Tiempo medio (s)",
    "ok_parse":  "Parse rate",
    "class_ok":  "Accuracy éxito/error",
    "similitud": "Similitud media",
    "key_match": "Cobertura campos",
})

# — redondear y mostrar —
summary = summary.round({
    "Tiempo medio (s)":      2,
    "Parse rate":            3,
    "Accuracy éxito/error":  3,
    "Similitud media":       3,
    "Cobertura campos":      3,
})

print("\n=== Resumen de evaluación y tiempos ===")
print(summary)

# — guardar CSV resumen —
summary.to_csv("resumen_evaluacion_modelos.csv")
