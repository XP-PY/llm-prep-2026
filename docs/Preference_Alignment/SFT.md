# Supervised Fine-Tuning (SFT) in Large Language Models

## Overview

Supervised Fine-Tuning (SFT) is the core **instruction-following** stage in the modern LLM training pipeline. After pre-training on massive unlabeled text with next-token prediction, the base model is further trained on a curated dataset of **high-quality (prompt, desired response)** pairs so that it learns to answer user requests in a more helpful, coherent, and consistently formatted way.

SFT bridges the gap between a powerful but "raw" autocomplete engine and a usable instruction-following assistant.

Key characteristics:
- **Supervised**: Uses explicit (prompt → response) pairs with cross-entropy loss on the desired tokens.
- **Fine-tuning**: Typically performed on the full model or with parameter-efficient methods ([LoRA](../PEFT/LoRA.md), [QLoRA](../PEFT/QLoRA.md), adapters).
- **Dataset size**: Usually 10k–200k high-quality examples (much smaller than pre-training data).
- **Goal**: Teach instruction following, style, format, safety patterns, and task-specific response behavior.

## Position in the Full Training Pipeline

| Stage              | Objective                          | Data Type                          | Typical Scale                  | Loss Function                  |
|--------------------|------------------------------------|------------------------------------|--------------------------------|--------------------------------|
| Pre-training       | Next-token prediction              | Unlabeled web text, books, code    | Trillions of tokens            | Causal language modeling       |
| **SFT**            | Instruction following              | Curated (prompt, response) pairs   | 10k–200k examples              | Causal LM on response tokens   |
| Preference Tuning (RLHF / DPO / KTO) | Alignment with human values       | Preference pairs or rankings       | 100k–1M preferences            | Reward modeling or direct pref |
| Post-training (safety, tools) | Additional capabilities & safeguards | Specialized datasets              | Varies                         | Varies                         |

SFT is the **foundation** for later alignment stages. If SFT is weak, later preference optimization usually has less to build on.

## What SFT Solves

SFT is mainly used when the problem is:

* the model does not follow instructions reliably,
* the output format is unstable,
* the tone or style is inconsistent,
* or the model needs to adapt to a domain-specific task pattern.

SFT is less about "which of several good answers is more preferred" and more about "can the model learn to answer in the demonstrated way at all?"

## Data Preparation

Quality > quantity. Common sources:

1. **Human-written or human-curated datasets**
   - ShareGPT-style conversation dumps
   - OpenAssistant
   - Dolly 15k

2. **Synthetic datasets**
   - Self-Instruct (Wang et al., 2022)
   - Alpaca
   - Evolution-based methods (e.g., UltraChat, Orca)
   - Distilled from stronger models

3. **Multimodal extensions**
   - LLaVA: (image, caption, detailed description) pairs
   - Qwen-VL: interleaved vision-language instructions

Best practices:
- Diversity in tasks (reasoning, coding, chat, safety)
- Clear formatting (e.g., `[INST] ... [/INST]` for Llama-style)
- Rejection of low-quality / toxic / off-topic responses
- Balanced length distribution
- Deduplication and contamination checks

## Training Details

### Objective
Standard next-token prediction, but **only compute loss on response tokens** (not on prompt tokens).

```math
\mathcal{L} = -\sum_{t \in \text{response}} \log p_\theta(y_t | y_{<t}, x)
```

where `x` is the prompt, `y` is the desired response.

### Hyperparameters (typical for 7B–70B models)
| Parameter              | Common Value                  | Notes                                      |
|------------------------|-------------------------------|--------------------------------------------|
| Learning rate          | 1e-5 – 2e-5                   | With warmup + cosine decay                 |
| Batch size (tokens)    | 2M–4M                         | Gradient accumulation if GPU memory limited|
| Epochs                 | 1–3                           | Avoid overfitting                          |
| Weight decay           | 0.1                           |                                            |
| LoRA rank              | 8–64                          | Higher rank → better but more params       |
| LoRA alpha             | 16–32                         |                                            |
| Target modules         | q_proj, k_proj, v_proj, o_proj| Sometimes gate_proj, up/down_proj in MLPs  |

### Parameter-Efficient Variants
- **Full fine-tuning**: All parameters updated (expensive, risk of catastrophic forgetting).
- **LoRA / QLoRA**: Low-rank adapters (r=16–64). QLoRA adds 4-bit quantization and can drastically reduce VRAM requirements.
- **DoRA**: Decomposed LoRA with magnitude/angle separation (slightly better than LoRA).
- **Adapters**: Bottleneck layers.

## Practical Implementation (Hugging Face Example)

```python
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_name = "meta-llama/Llama-2-7b-hf"  # or "Qwen/Qwen2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Example dataset: Alpaca
dataset = load_dataset("yahma/alpaca-cleaned")

def formatting_prompts_func(example):
    return {"text": f"[INST] {example['instruction']} [/INST] {example['output']}"}

dataset = dataset.map(formatting_prompts_func)

# Tokenize with loss only on response
def tokenize_func(example):
    tokenized = tokenizer(example["text"], truncation=True, max_length=1024)
    # Find response start
    inst_end = example["text"].find("[/INST]") + len("[/INST]")
    response_start_idx = len(tokenizer(example["text"][:inst_end])["input_ids"])
    labels = tokenized["input_ids"].copy()
    labels[:response_start_idx] = -100  # Ignore prompt tokens
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_func, remove_columns=dataset.column_names)

# LoRA config
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./llama2-7b-alpaca-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit",  # for QLoRA
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()
```

## Common Pitfalls & Mitigations

| Issue                          | Symptom                              | Fix                                      |
|--------------------------------|--------------------------------------|------------------------------------------|
| Overfitting                    | Great on train, poor generalization  | Early stopping, more diverse data        |
| Catastrophic forgetting        | Loss of general knowledge            | LoRA/QLoRA, continued pre-training mix  |
| Repetition / verbosity         | Model loops or rambles               | Clean data, length normalization         |
| Prompt format sensitivity      | Needs exact tags                     | Consistent formatting during training    |
| Contamination                  | Leaks eval benchmarks                | Careful dataset curation & checks        |

## Real-World Examples

- **Llama-2-Chat**: Strong capability jump over the base model after instruction tuning.
- **Llama-3 / Llama-3.1 Instruct**: Heavy use of high-quality synthetic and curated instruction data in post-training.
- **Mistral / Mixtral Instruct**: Strong instruction-tuned variants built with efficient SFT and lightweight alignment.
- **Many vertical models**: Domain assistants in law, customer service, coding, or finance often start by using SFT to lock in task format and tone before trying more complex preference optimization.
