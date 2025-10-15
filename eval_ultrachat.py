import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

model_id = 'Qwen3-4B'
YOUR_HF_USERNAME = os.environ['HF_USERNAME']

new_model_id = output_dir = f"{YOUR_HF_USERNAME}/{model_id.rsplit('/')[-1]}-qlora-ultrachat"

tokenizer = AutoTokenizer.from_pretrained(new_model_id)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    # new_model_id,
    output_dir + "/checkpoint-10",
    # output_dir + "/checkpoint-30",
    # output_dir + "/checkpoint-100",
    quantization_config=quantization_config,
    device_map="auto",
)

text = "### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant:"

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

print("After attaching Lora adapters:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

# %%
# Test model behavior without LoRA
model.disable_adapters()
outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

print("Before Lora:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
