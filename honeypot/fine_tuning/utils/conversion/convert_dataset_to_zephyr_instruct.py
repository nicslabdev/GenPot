import re

# Lee el dataset original
with open('llama3_synology_dataset.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# Busca pares user/assistant
samples = re.findall(
    r"<\|start_header_id\|>user\n(.*?)<\|end_header_id\|>\n<\|start_header_id\|>assistant\n(.*?)<\|end_header_id\|>",
    data, re.DOTALL
)

# Convierte al formato Zephyr
with open('zephyr_synology_dataset.txt', 'w', encoding='utf-8') as f:
    for user, assistant in samples:
        f.write(f"[INST] {user.strip()} [/INST]\n{assistant.strip()}\n\n")