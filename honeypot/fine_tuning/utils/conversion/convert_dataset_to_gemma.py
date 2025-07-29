import re
import os
from dotenv import load_dotenv
load_dotenv()

llama3_dataset_path = os.getenv("DATASETS_BASE_DIR") + os.getenv("LLAMA_DATASET_NAME")
gemma_dataset_path = os.getenv("DATASETS_BASE_DIR") + os.getenv("GEMMA_DATASET_NAME")

# Lee el dataset LLaMA 3
with open(llama3_dataset_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Extrae pares user/assistant
samples = re.findall(
    r"<\|start_header_id\|>user\n(.*?)<\|end_header_id\|>\n<\|start_header_id\|>assistant\n(.*?)<\|end_header_id\|>",
    data, re.DOTALL
)

# Convierte a formato Gemma instruct
with open(gemma_dataset_path, 'w', encoding='utf-8') as f:
    for user, assistant in samples:
        user = user.strip()
        assistant = assistant.strip()
        f.write(f"<start_of_turn>user\n{user}\n<end_of_turn>\n<start_of_turn>model\n{assistant}\n<end_of_turn>\n\n")
