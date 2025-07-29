#Script que comprueba para cada modelo si ha generado un json coherente con la estructura que debe tener para la api de synology
import os
import json
import re
from expected_structures import validate_response_structure

MODELS = ["llama3", "zephyr", "gemma"]
RESULTS_DIR = "results"

def parse_slug(fname):
    # Convierte el slug en API, method y version 
    # Ejemplo: 2_api_SYNO_FileStation_List_method_list_share_version_2.json
    m = re.search(
        r'api_([A-Za-z0-9_]+)_method_([A-Za-z0-9_]+)_version_([0-9]+)',
        fname
    )
    if not m:
        return "", "", ""
    api = m.group(1).replace("_", ".")  # convierte SYNO_FileStation_List -> SYNO.FileStation.List
    method = m.group(2)
    version = m.group(3)
    return api, method, version

def main():
    for model in MODELS:
        model_dir = os.path.join(RESULTS_DIR, model)
        print(f"\nEvaluando modelo: {model}")
        total = 0
        json_valid = 0
        structurally_ok = 0

        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(model_dir, fname)
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            total += 1
            try:
                data = json.loads(raw)
                json_valid += 1
            except Exception:
                print(f"{fname}: ❌ NO ES JSON VÁLIDO")
                continue
            api, method, version = parse_slug(fname)
            if not (api and method and version):
                print(f"{fname}: ⚠️  No se pudo parsear endpoint.")
                continue
            if validate_response_structure(data, api, method, version):
                structurally_ok += 1
                print(f"{fname}: ✔️ OK estructura")
            else:
                print(f"{fname}: ❌ Mal estructurado")

        print(f"\nResumen modelo {model}:")
        print(f"  Total pruebas: {total}")
        print(f"  JSON válidos: {json_valid}")
        print(f"  Estructura correcta: {structurally_ok}")
        print(f"  Porcentaje estructuralmente OK: {structurally_ok/total*100:.1f}%\n")

if __name__ == "__main__":
    main()
