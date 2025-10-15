# Original colab notebook: https://colab.research.google.com/drive/1vIjBtePIZwUaHWfjfNHzBjwuXOyU_ugD?usp=sharing#scrollTo=QeU992x4zN2Z

# %%
# Need to log into huggingface to access emta-llama/Llama-2-7b-hf which is gated.
# Alternatively, one may do "hf auth login" from the command line to log in.
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    login(token=token, add_to_git_credential=True)
else:
    raise RuntimeError(
        "Set the HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variable, "
        "or run `hf auth login` once before executing this script."
    )

# %%
# Load the model and tokenizer

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback

# Load the 7b llama model
model_id = "meta-llama/Llama-2-7b-hf"  # requires separate authorization
# model_id = "Qwen/Qwen3-4B"
# model_id = "Qwen/Qwen3-4B-FP8"

config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    # We set attn_implementation explicitly because
    #  1. both Qwen3-4B and Llama-2-7b-hf do not set attn_implementation (config._attn_implementation is None)
    #  2. We set packing=True in the trainer config and the default attn_implementation=eager does not work.
    attn_implementation="flash_attention_2"
)

# Inspect the model config to decide whether to apply 4-bit quantization locally.
use_quantization = True
# use_quantization = False
if use_quantization and getattr(config, "quantization_config", None) is None:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
else:
    # Pre-quantized checkpoints (e.g. FP8) already provide a quantization config.
    quantization_config = None

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    quantization_config=quantization_config,
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    # Note: older models like Llama2 doesn't have a pad_token.
    # Set it to a new token to correctly attend to EOS tokens.
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
print(f'tokenizer pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}')

# %%
# LoRA config
from peft import LoraConfig

# lora_rank = 1
lora_rank = 8

lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=32,
    lora_dropout=0.2,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(lora_config, adapter_name="default")
# 17M trainable parameters for model=Qwen3-4B with (r=8, target_modules=(7))
# 2M trainable parameters for model=Qwen3-4B with (r=1, target_modules=(7))
print(f'Number of trainable params={(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6):.0f}M')

# %%
# Load the data
from datasets import load_dataset

# UltraChat contains multi-round dialogue created using ChatGPT Turbo APIs.
# https://huggingface.co/datasets/stingning/ultrachat
# Here we only load 1% of the training data (1.468M -> 14.68K).
train_dataset = load_dataset("stingning/ultrachat", split="train[:1%]")

# Example:
# train_dataset[0]
# {'id': '0',
#  'data': ['How can cross training benefit groups like runners, swimmers, or weightlifters?',
#   'Cross training can benefit groups like runners, swimmers, or weightlifters in the following ways:\n\n1. Reduces the risk of injury: Cross training involves different types of exercises that work different muscle groups. This reduces the risk of overuse injuries that may result from repetitive use of the same muscles.\n\n2. Improves overall fitness: Cross training helps improve overall fitness levels by maintaining a balance of strength, endurance, flexibility, and cardiovascular fitness.\n\n3. Breaks monotony: Cross training adds variety to your fitness routine by introducing new exercises, which can help you stay motivated and avoid boredom that often comes with doing the same exercises repeatedly.\n\n4. Increases strength: Cross training helps in building strength by incorporating exercises that target different muscle groups. This helps you build strength in areas that may be underdeveloped.\n\n5. Enhances performance: Cross training allows you to work on different aspects of fitness that are essential for your sport or activity. For example, a runner can benefit from strength training as it helps build stronger muscles and improves running economy.\n\nOverall, cross training offers numerous benefits to athletes and fitness enthusiasts. By incorporating other forms of exercise into their routine, individuals can improve their overall fitness, reduce the risk of injury, and enhance their performance.',
#   "That makes sense. I've been wanting to improve my running time, but I never thought about incorporating strength training. Do you have any recommendations for specific exercises?",
#   "Sure, here are some strength training exercises that can benefit runners:\n\n1. Squats: Squats target the glutes, quadriceps, and hamstrings. They help improve lower-body strength, power, and stability, making them an excellent exercise for runners.\n\n2. Lunges: Lunges target the same muscles as squats but also work the hip flexors and help improve balance.\n\n3. Deadlifts: Deadlifts are a compound exercise that targets the glutes, hamstrings, and lower back. They improve lower body strength, power, and stability.\n\n4. Plyometric exercises: Plyometric exercises such as jump squats, box jumps, or single-leg hops can help improve explosive power, which is crucial for sprinting.\n\n5. Calf raises: Calf raises target the calves and help improve running economy by strengthening the muscles that propel you forward.\n\nIt's important to remember to start with lighter weights and proper form to avoid injury. I recommend consulting with a personal trainer or coach for guidance on proper form and technique.",
#   "Hmm, I'm not really a fan of weightlifting though. Can I incorporate other forms of exercise into my routine to improve my running time?",
#   "Yes, absolutely! In addition to strength training, there are many other types of exercises that can help improve running performance without involving weightlifting. Here are some examples:\n\n1. Plyometric exercises: Plyometric exercises like jump squats, box jumps or single-leg jumps improve explosive power which can make your runs more efficient.\n\n2. Hill training: Hill training is a great way to improve your overall stamina, strength, and speed.\n\n3. Circuit training: Circuit training is a full-body workout that can help build endurance, strength, and agility. You can include exercises like push-ups, lunges, burpees, and jump ropes to make the circuit more challenging.\n\n4. Yoga: Yoga can help improve running performance by increasing flexibility, strength, and balance. It's also a great way to reduce stress and stay focused.\n\n5. Swimming or cycling: Swimming or cycling are low-impact exercises that provide a great cardiovascular workout, which can help improve endurance and fitness.\n\nRemember to always listen to your body and give yourself adequate rest and recovery time between workouts. Mix up your workouts to keep things interesting and challenge your body in new ways."]}

# %%
# Set up training config
from trl.trainer.sft_config import SFTConfig

YOUR_HF_USERNAME = os.environ['HF_USERNAME']

new_model_id = output_dir = f"{YOUR_HF_USERNAME}/{model_id.rsplit('/')[-1]}-qlora-ultrachat"
# Note: with (batch_size=4, grad_acc_steps=4), training is already compute-bound, so further
# # increasing batch_size won't speedup training. ~8s/step.
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
# per_device_train_batch_size = 8
# gradient_accumulation_steps = 2
optim = "paged_adamw_32bit"
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
    push_to_hub=False,
    # push_to_hub=True,
    # Optimization setup
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    max_grad_norm=max_grad_norm,
    gradient_checkpointing=True,
    # Learning rate
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    # Packing related
    packing=True,  # this also implies padding_free=True in SFTTrainer.__init__
    packing_strategy="bfd",  # default
    max_length=1024,
    # Note "max_steps" overrides "max_train_epochs".
    max_steps=max_steps,
    # TBA
    # dataset_text_field="id",
    # dataset_text_field="data",
)


# %%
# Set up trainer configs
from trl.trainer.sft_trainer import SFTTrainer

def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text


class SampleGenerationCallback(TrainerCallback):
    """Periodically generate model outputs for fixed prompts during training."""

    def __init__(self, tokenizer, prompts, max_new_tokens=200, sample_steps=50):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.sample_steps = sample_steps

    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if self.sample_steps is None or state.global_step % self.sample_steps != 0:
            return
        if not state.is_local_process_zero:
            return
        if model is None:
            return

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        for prompt in self.prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            try:
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                print(
                    f"\n[step {state.global_step}] prompt: {prompt}\n[step {state.global_step}] output: {decoded}\n",
                    flush=True,
                )
            except RuntimeError as err:
                print(f"[step {state.global_step}] Generation failed: {err}", flush=True)
                break

        if was_training:
            model.train()

eval_prompt = "### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant:"
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
    callbacks=[
        SampleGenerationCallback(
            tokenizer,
            prompts=[eval_prompt],
            max_new_tokens=200,
            sample_steps=logging_steps,  # note dense sampling might lead to low GPU utilization
        )
    ],
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
    project="PEFT examples",
    # mode="disabled",
    # dir=wandb_dir,  # wandb output is written to this directory
    name=run_name,  # run name (used in wandb GUI)
    # Note: no need to provide config to wandb.init as SFTTrainer already passes the config to wandb.
    # config={
    #     "quantization": asdict(quantization_config) if quantization_config is not None else None,
    #     "trainer": asdict(training_arguments),
    #     "lora": asdict(lora_config),
    # },
)

try:
    trainer.train()
except KeyboardInterrupt:
    # Ensure a clean wandb exit, otherwise one might get "Broken Pipe" error.
    run.finish(exit_code=1)
    raise
else:
    run.finish()

# %%
# Load the model for evaluation
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# tokenizer = AutoTokenizer.from_pretrained(new_model_id)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
# )

model = AutoModelForCausalLM.from_pretrained(
    # new_model_id,
    # output_dir + "/checkpoint-30",
    output_dir + "/checkpoint-100",
    quantization_config=quantization_config,
    # adapter_kwargs={"revision": "09487e6ffdcc75838b10b6138b6149c36183164e"}
    device_map="auto",
)

# %%
# Test model behavior (with LoRA)
text = eval_prompt

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,  # need to provide attention mask explicitly
    max_new_tokens=250,
    do_sample=False,  # greedy decoding with do_sample=False
)

print("After attaching Lora adapters:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

# %%
# Test model behavior without LoRA
model.disable_adapters()
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,  # need to provide attention mask explicitly
    max_new_tokens=250,
    do_sample=False,  # greedy decoding with do_sample=False
)

print("Before Lora:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
