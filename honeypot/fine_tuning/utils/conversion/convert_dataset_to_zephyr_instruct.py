import re
import os
from dotenv import load_dotenv
load_dotenv()

llama3_dataset_path = os.getenv("DATASETS_BASE_DIR") + os.getenv("LLAMA_DATASET_NAME")
zephyr_dataset_path = os.getenv("DATASETS_BASE_DIR") + os.getenv("ZEPHYR_DATASET_NAME")

# Lee el dataset original
with open(llama3_dataset_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Busca pares user/assistant
samples = re.findall(
    r"<\|start_header_id\|>user\n(.*?)<\|end_header_id\|>\n<\|start_header_id\|>assistant\n(.*?)<\|end_header_id\|>",
    data, re.DOTALL
)

# Convierte al formato Zephyr
with open(zephyr_dataset_path, 'w', encoding='utf-8') as f:
    for user, assistant in samples:
        f.write(f"[INST] {user.strip()} [/INST]\n{assistant.strip()}\n\n")