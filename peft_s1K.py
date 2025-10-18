# Finetune model on s1K to evaluate simple test-time scaling proposed in https://arxiv.org/pdf/2501.19393

# %%
# Load the model and tokenizer

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback

# Load the base model
model_id = "Qwen/Qwen3-4B-Instruct-2507"

config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)

# Inspect the model config to decide whether to apply 4-bit quantization locally.
use_quantization = True
# use_quantization = False
if use_quantization and getattr(config, "quantization_config", None) is None:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # QLoRA requires the following 2 params
        # See https://huggingface.co/docs/transformers/en/quantization/bitsandbytes?bnb=4-bit#qlora
        bnb_4bit_compute_dtype=torch.float16,  # default=torch.float32
        bnb_4bit_quant_type="nf4",
    )
else:
    # Pre-quantized checkpoints (e.g. Qwen3-4B-FP8) already provide a quantization config.
    quantization_config = None

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    dtype="auto",
    # Note: quantized params are NOT trainable.
    quantization_config=quantization_config,
    trust_remote_code=True,
)

# Check model memory usage
print(f"Model memory footprint: {model.get_memory_footprint()/1000**3:.2f}GB")

# Inspect quantization
for name, p in model.named_parameters():
    if 'layers.0.' in name:
        print(f'{name}, dtype={p.dtype}, shape={p.shape}')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
assert tokenizer.pad_token_id != tokenizer.eos_token_id
print(f'tokenizer pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}')

# %%
# LoRA config
from peft import LoraConfig

lora_rank = 1
# lora_rank = 8

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=32,
    lora_dropout=0.2,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(lora_config, adapter_name="default")
# Note: p.numel() uses 0.5 for 4-bit params.
# 17M trainable parameters for model=Qwen3-4B with (r=8, target_modules=(7))
# 2M trainable parameters for model=Qwen3-4B with (r=1, target_modules=(7))
print(f'Number of trainable params={(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.0f}M, total params={(sum(p.numel() for p in model.parameters()) / 1e6):.0f}M')

# %%
# Load the data
from datasets import load_dataset

dataset = load_dataset("simplescaling/s1K_tokenized")
train_dataset = dataset["train"]

# %%
# Set up training config
from trl.trainer.sft_config import SFTConfig

output_dir = f"ckpts/{model_id.rsplit('/')[-1]}-qlora-s1K"
push_to_hub = False
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"  # standard companion with 4-bit QLoRA training
save_steps = 10
# logging_steps = 5  # dense logging for debugging, but this would slow down training
logging_steps = 10
# Learning rate
# Note: lr=2e-4 leads to training error blow up after 30 steps.
learning_rate = 2e-4
# learning_rate = 2e-5
warmup_ratio = 0.03
# lr_scheduler_type = "constant_with_warmup"
lr_scheduler_type = "cosine"
max_grad_norm = 0.3
# Number of batch updates / optimizer updates, which is gradient_accumulation_steps * minibatch updates.
max_steps = 100

training_arguments = SFTConfig(
    # Model saving/logging
    output_dir=output_dir,
    save_steps=save_steps,
    logging_steps=logging_steps,
    push_to_hub=push_to_hub,
    # Optimization setup
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    max_grad_norm=max_grad_norm,
    # gradient_checkpointing=False,  # fills the entire VRAM=32GB
    gradient_checkpointing=True,  # default
    # Learning rate
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    # Packing related
    packing=True,  # this also implies padding_free=True in SFTTrainer.__init__
    packing_strategy="bfd",  # default
    max_length=1024,
    # # Chat template
    # assistant_only_loss=True,  # only works with formatting_func2.
    # Note "max_steps" overrides "max_train_epochs".
    max_steps=max_steps,
    # Note: we don't need dataset_text_field as we pass formatting_func to SFTTrainer.
    # dataset_text_field="id",
    # dataset_text_field="data",
    dataset_text_field="text",  # default
)


# %%
# Set up trainer configs

instruction_template = "<|im_start|>user"
response_template = "<|im_start|>assistant\n"
# Use a token that is never used
tokenizer.pad_token = "<|fim_pad|>"

# from trl.trainer import DataCollatorForCompletionOnlyLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False
)

from trl.trainer.sft_trainer import SFTTrainer

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    args=training_arguments,
    data_collator=collator
)

# %%
# Set up wandb config
import time
import wandb
from datetime import datetime
from dataclasses import asdict

start_time = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
run_name = (
    model.config._name_or_path
    + f"_{train_dataset.info.dataset_name}"
    + f"_bs{per_device_train_batch_size}"
    + f"_lr{learning_rate}"
    + f"_{start_time}"
)

# %%
# Train the model

run = wandb.init(
    project="PEFT on s1K",
    # mode="disabled",
    name=run_name,  # run name (used in wandb GUI)
)

try:
    trainer.train()
except KeyboardInterrupt:
    # Ensure a clean wandb exit, otherwise one might get "Broken Pipe" error.
    run.finish(exit_code=1)
    raise
except Exception:
    run.finish(exit_code=1)
    raise
else:
    run.finish(exit_code=0)