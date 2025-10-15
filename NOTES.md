A place to take notes on iterations.

#### v1
First, in `peft_ultrachat.py`, we do the following:
 1. Quantize a `Qwen3-4B` model to `nf4`.
 2. Use LoRA to fine-tune 7 types of weight matrices with `r=1`.
 3. Train it on `ultrachat` data for 100 steps, which is less than one epoch of the data.
After 100 steps, training loss reaches `1.187`, VRAM usage is `7GB` per `wandb`'s `GPU Memory Allocated (Bytes)` record. Training time is `7.5s/step`.

#### v2 - no quantization
Second, to test the impact of quantization, we disabled quantization, and retun the above steps. Training time is still `7.5s/step` (or `13min` for `100` steps), but VRAM usage increased to `28GB`.

Also, we noticed v2 has higher accuracy and lower loss than v1, likely caused by the higher decision of the full model (`bf16` vs `nf4`)
![alt text](assets/img1_loss_v2_vs_v1.png)

#### v3 - Llama-2-7b-hf
From v1, we switch from `Qwen3-4B` to `Llama-2-7b-hf`, a big yet older model. We see significant train loss reduction from `1.1871` to `0.9023`.

`Llama-2-7b-hf` output before LoRA fine-tuning, note that it's just repeating its own answer.
```
<s> ### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant: Sure. nobody can explain it in simple terms.### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant: Sure. nobody can explain it in simple terms.

### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant: Sure. nobody can explain it in simple terms.

### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant: Sure. nobody can explain it in simple terms.

### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant: Sure. nobody can explain it in simple terms.

### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant: Sure. nobody can explain it in simple terms.

### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML
```

`Llama2-7b-hf` after fine-tuning (??? steps)
TBA

`Qwen3-4B` output before LoRA fine-tuning:
```
### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant:### 
Okay, the user is asking for an explanation of contrastive learning in machine learning, and they want it simple for someone new to the field. Let me start by recalling what contrastive learning is. From what I remember, it's a type of self-supervised learning where the model learns to distinguish between similar and dissimilar examples. But I need to break this down into simpler terms.

First, I should define what self-supervised learning is, maybe. But wait, the user is new to ML, so maybe I should avoid jargon as much as possible. Let me think of an analogy. Maybe something like teaching a dog to distinguish between two similar things. Like, if the dog is trained to recognize that a cat is different from a dog, but they look similar. But how does that relate to contrastive learning?

Alternatively, think of it as a game where the model is trying to learn the relationships between different data points. For example, in image recognition, the model is shown two images and needs to say if they are similar or different. But how does that translate into the algorithm?

Wait, contrastive learning is often used in embedding spaces. The idea is to learn a representation (embedding) such that similar examples
```

#### Training stats
| Model       | Quantization | Loss   | Runtime | VRAM    |
|-------------|--------------|--------|---------|---------|
| Qwen3-4B    | Y            | 1.1871 |  7.5s/step |  6.5GB  |
| Qwen3-4B    | N            | 1.1562 |  7.5s/step  | 29.4GB  |
| Llama-2-7b-hf  | Y         | 0.9023 |  9.9s/step  | 11.5GB  |
