from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_path = "./models/zephyr_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, model_path)

prompt = '[INST] Return valid JSON only. API: SYNO.Core.User Method: list Params: {"version": 1} [/INST]'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

output = model.generate(
    input_ids=input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
