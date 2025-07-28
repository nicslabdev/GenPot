#generate dataset for fine tunning gemma model
import random
import json
from faker import Faker

fake = Faker()

apis = [
    ("SYNO.API.Auth", "login", {"version": 6, "account": "", "passwd": ""}),
    ("SYNO.Core.System", "info", {"version": 1}),
    ("SYNO.FileStation.List", "list_share", {"version": 2}),
    ("SYNO.FileStation.List", "list", {"version": 2, "folder_path": ""}),
    ("SYNO.FileStation.Download", "download", {"version": 2, "path": ""}),
    ("SYNO.DownloadStation.Task", "list", {"version": 1}),
]

def generate_example():
    api, method, params = random.choice(apis)
    if "account" in params:
        params["account"] = fake.email()
        params["passwd"] = fake.password()
    if "folder_path" in params:
        folder = random.choice(["projects", "marketing", "hr", "finance", "engineering"])
        params["folder_path"] = f"/volume1/{folder}"
    if "path" in params:
        folder = random.choice(["hr", "finance", "docs"])
        filename = fake.file_name(extension="xlsx")
        params["path"] = f"/volume1/{folder}/{filename}"
    if "version" in params:
        params["version"] = random.choice([1, 2, 3])

    if random.random() < 0.7:
        if method == "list_share":
            response = {
                "success": True,
                "data": {
                    "shares": [
                        {"name": "Public", "path": "/volume1/public"},
                        {"name": "HR", "path": "/volume1/hr"},
                        {"name": "Engineering", "path": "/volume1/engineering"},
                    ]
                }
            }
        elif method == "list":
            response = {
                "success": True,
                "data": {
                    "files": [
                        {"name": fake.file_name(), "size": random.randint(10000, 500000)},
                        {"name": fake.file_name(), "size": random.randint(10000, 500000)}
                    ]
                }
            }
        elif method == "info":
            response = {
                "success": True,
                "data": {
                    "model": "RS" + str(random.randint(800, 1000)) + "+",
                    "serial": fake.bothify(text="????####"),
                    "firmware": "DSM 7.1." + str(random.randint(0, 5))
                }
            }
        elif method == "login":
            response = {
                "success": True,
                "data": {
                    "sid": fake.uuid4()
                }
            }
        elif method == "download":
            response = {
                "success": True,
                "data": {
                    "task": "download_" + fake.uuid4()
                }
            }
        else:
            response = {"success": True, "data": {}}
    else:
        response = {
            "success": False,
            "error": {
                "code": random.choice([101, 400, 403, 105]),
                "message": random.choice(["Invalid parameter", "Access denied", "Session timeout", "Unknown error"])
            }
        }

    user_block = f"<|begin_of_text|><|start_header_id|>user\nAPI: {api}\nMethod: {method}\nParams: {json.dumps(params)}\n<|end_header_id|>"
    assistant_block = f"<|start_header_id|>assistant\n{json.dumps(response, indent=2)}\n<|end_header_id|>"

    return f"{user_block}\n{assistant_block}"

# Generar 100 ejemplos
examples = [generate_example() for _ in range(100)]

# Guardar en archivo
import os
output_path = "synology_synthetic_dataset_gemma.txt"
with open(output_path, "w") as f:
    f.write("\n".join(examples))

output_path
