import random
import json
from pathlib import Path

# Carpetas empresariales realistas
departments = ["Finance", "HR", "Legal", "Marketing", "Sales", "IT", "Compliance", "Engineering", "Logistics"]
subfolders = ["Reports", "Contracts", "Policies", "Plans", "Archive", "Projects", "Backups"]
filenames = ["budget.xlsx", "employee_list.pdf", "nda.docx", "security_policy.pdf", "roadmap_2024.txt", "invoice_1032.pdf"]

# Métodos DSM y sus estructuras
api_templates = [
    ("SYNO.FileStation.List", "list", lambda: {"folder_path": f"/volume1/{random.choice(departments)}/{random.choice(subfolders)}"}),
    ("SYNO.FileStation.Create", "create", lambda: {"folder_path": f"/volume1/{random.choice(departments)}/NewFolder{random.randint(1, 10)}"}),
    ("SYNO.FileStation.Delete", "delete", lambda: {"path": f"/volume1/{random.choice(departments)}/{random.choice(filenames)}"}),
    ("SYNO.FileStation.Rename", "rename", lambda: {
        "path": f"/volume1/{random.choice(departments)}/draft_{random.randint(1,9)}.txt",
        "name": f"final_{random.randint(1,9)}.txt"
    }),
    ("SYNO.FileStation.Move", "move", lambda: {
        "from": f"/volume1/{random.choice(departments)}/Temp/{random.choice(filenames)}",
        "to": f"/volume1/{random.choice(departments)}/{random.choice(subfolders)}/"
    }),
    ("SYNO.FileStation.Copy", "copy", lambda: {
        "from": f"/volume1/{random.choice(departments)}/Manuals/{random.choice(filenames)}",
        "to": f"/volume1/{random.choice(departments)}/Backup/"
    }),
    ("SYNO.FileStation.Search", "start", lambda: {
        "folder_path": f"/volume1/{random.choice(departments)}/{random.choice(subfolders)}",
        "pattern": random.choice(["*.pdf", "*.docx", "*.txt"])
    }),
    ("SYNO.Core.System", "get", lambda: {}),
    ("SYNO.Core.SyslogClient", "list", lambda: {})
]

# Respuesta simulada por defecto para el entrenamiento
fake_response = {"status": "success", "data": {"info": "Example response."}}

# Generar ejemplos
examples = []
for _ in range(500):
    api, method, param_func = random.choice(api_templates)
    example = {
        "api": api,
        "method": method,
        "params": param_func(),
        "response": fake_response
    }
    examples.append(example)

# Guardar en archivo JSONL
output_path = Path("synology_dataset_generated.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for e in examples:
        f.write(json.dumps(e) + "\n")

print(f"✅ Dataset generado con 500 ejemplos realistas en: {output_path}")
