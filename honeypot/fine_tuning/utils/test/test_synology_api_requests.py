import requests

# URL del honeypot que simula la API Synology
BASE_URL = "http://localhost:80/webapi/entry.cgi"

# Lista de peticiones de prueba (simulando a un atacante)
requests_to_test = [
    {"api": "SYNO.Core.System", "method": "info", "version": 1},
    {"api": "SYNO.Core.User", "method": "list", "version": 1},
    {"api": "SYNO.FileStation.List", "method": "list_share", "version": 2},
    {"api": "SYNO.DownloadStation.Task", "method": "list", "version": 1},
    {"api": "SYNO.Core.Security", "method": "scan", "version": 1},
]

print("🚀 Enviando peticiones POST a Synology Honeypot...\n")

for test in requests_to_test:
    try:
        response = requests.post(BASE_URL, data=test, timeout=60)
        print(f"✅ {test['api']} / {test['method']} => HTTP {response.status_code}")
        print("Respuesta:")
        print(response.text)
        print("-" * 80)
    except Exception as e:
        print(f"❌ Error en {test['api']} / {test['method']}: {e}")
