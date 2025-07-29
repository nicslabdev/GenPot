import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

DATASET_PATH = os.getenv("DATASETS_BASE_DIR") + os.getenv("LLAMA_DATASET_NAME")
EXPECTED_DIR = "expected_responses"
os.makedirs(EXPECTED_DIR, exist_ok=True)

def extract_api_method_version(user_prompt):
    """Extrae api, method y version del prompt de usuario."""
    api_match = re.search(r'API:\s*([^\n]+)', user_prompt)
    method_match = re.search(r'Method:\s*([^\n]+)', user_prompt)
    params_match = re.search(r'Params:\s*({.*})', user_prompt)
    api = api_match.group(1).strip() if api_match else ""
    method = method_match.group(1).strip() if method_match else ""
    version = ""
    if params_match:
        try:
            params = json.loads(params_match.group(1))
            version = str(params.get("version", ""))
        except Exception:
            pass
    return api, method, version

def slugify(query):
    return re.sub(r'[^a-zA-Z0-9]', '_', query)

with open(DATASET_PATH, encoding="utf-8") as f:
    raw = f.read()

pairs = re.findall(
    r"<\|start_header_id\|>user\n(.*?)<\|end_header_id\|>\n<\|start_header_id\|>assistant\n(.*?)<\|end_header_id\|>",
    raw, re.DOTALL
)

# Indexa el dataset por (api, method, version)
index = {}
for user, assistant in pairs:
    api, method, version = extract_api_method_version(user)
    key = (api, method, version)
    index[key] = assistant.strip()

# Define tus peticiones de test (las del bash script)
test_requests = [
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=password123",
  "api=SYNO.FileStation.List&method=list_share&version=2",
  "api=SYNO.Core.System&method=info&version=1",
  "api=SYNO.DownloadStation.Task&method=list&version=1",
  "api=SYNO.FileStation.List&method=list&version=2&folder_path=/volume1",
  "api=SYNO.FileStation.Info&method=get&version=2&path=/volume1/public",
  "api=SYNO.Core.User&method=get&version=1&user_name=admin",
  "api=SYNO.Core.User&method=logout&version=1&user_name=admin",
  "api=SYNO.Core.System.Utilization&method=get&version=1",
  "api=SYNO.Core.System.Status&method=network_status&version=1",
  "api=SYNO.API.Auth&method=login&version=6&account=admin&passwd=1234",
  "api=SYNO.API.Auth&method=logout&version=6&session=DownloadStation",
  "api=SYNO.DownloadStation.Task&method=create&version=1&uri=http://example.com/file.iso",
  "api=SYNO.DownloadStation.Task&method=delete&version=1&id=taskid_123",
  "api=SYNO.Core.Network&method=list&version=1",
  "api=SYNO.Core.Storage.Volume&method=status&version=1",
  "api=SYNO.Core.ExternalDevice&method=list&version=1",
  "api=SYNO.Core.Time&method=get&version=1",
  "api=SYNO.FakeModule.Bogus&method=nonexistent&version=1",
  #Algunas peticiones erroneas para saber su comportamiento ante errores
  "api=SYNO.Core.System&method=info&version=999",   # versión inválida
  "api=SYNO.Core.User&method=delete&version=1&user_name=",   # parámetro vacío
  "api=SYNO.Core.User&metod=login&version=1"   # typo en 'method'
]

for i, req in enumerate(test_requests, 1):
    slug = f"{i}_{slugify(req)}"
    # Extrae los campos de la query
    api = ""
    method = ""
    version = ""
    for part in req.split("&"):
        if part.startswith("api="): api = part.split("=")[1]
        if part.startswith("method="): method = part.split("=")[1]
        if part.startswith("version="): version = part.split("=")[1]
    key = (api, method, version)
    # Busca el ejemplo en el dataset
    best = index.get(key, None)
    # Si no lo encuentra, pon ejemplo estándar de error
    if best is None:
        best = json.dumps({
            "success": False,
            "error": {"code": 1001, "message": "Unknown API or method"}
        }, indent=2)
    fname = os.path.join(EXPECTED_DIR, f"{slug}.json")
    with open(fname, "w") as f:
        f.write(best)

print("Archivos expected responses generados en", EXPECTED_DIR)
