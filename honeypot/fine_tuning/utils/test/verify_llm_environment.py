import os
import requests
import json

PROMPT_PATH = "/mnt/AI-DATA/jzamoraru_tfm/prompts/gemma.txt"
MODEL_NAME = "gemma"
FASTAPI_URL = "http://localhost:8000/generate"

# Paso 1: Verificar el archivo de plantilla
print("🔍 Verificando archivo de prompt...")

if not os.path.exists(PROMPT_PATH):
    print(f"❌ Archivo no encontrado: {PROMPT_PATH}")
    exit(1)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    template = f.read().strip()

if not template:
    print(f"⚠️ El archivo {PROMPT_PATH} está vacío.")
    exit(1)

print("✅ Archivo de prompt cargado correctamente.")

# Paso 2: Construir un prompt de prueba
prompt = template.format(
    api="SYNO.Core.System",
    method="info",
    params=json.dumps({"version": 1}, ensure_ascii=False)
)

print("📤 Prompt generado:")
print(prompt)

# Paso 3: Verificar conexión a FastAPI
print("\n🔍 Verificando conexión con FastAPI...")

try:
    response = requests.post(
        FASTAPI_URL,
        json={"prompt": prompt, "model": MODEL_NAME},
        timeout=60
    )
except requests.exceptions.ConnectionError:
    print("❌ No se pudo conectar a FastAPI en localhost:8000.")
    exit(1)

# Paso 4: Comprobar respuesta
if response.status_code != 200:
    print(f"❌ Error en la respuesta del servidor: {response.status_code}")
    print(response.text)
    exit(1)

try:
    data = response.json()
except json.JSONDecodeError:
    print("❌ La respuesta no es JSON válido:")
    print(response.text)
    exit(1)

if "parsed" in data:
    print("✅ Respuesta JSON interpretada correctamente:")
    print(json.dumps(data["parsed"], indent=2))
elif "raw" in data:
    print("⚠️ Solo se recibió respuesta en bruto:")
    print(data["raw"])
else:
    print("❌ La respuesta no contiene campos esperados ('parsed' o 'raw'):")
    print(data)
    exit(1)

print("\n✅ Verificación completada con éxito. El entorno LLM está listo.")
