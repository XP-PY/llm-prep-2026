# Supervised Fine-Tuning (SFT) in Large Language Models

## Overview

Supervised Fine-Tuning (SFT) is the critical **instruction-following** stage in the modern LLM training pipeline. After pre-training on massive unlabeled text (next-token prediction), the base model is further trained on a curated dataset of **high-quality (input, desired output)** pairs so that it learns to generate helpful, coherent, and styled responses when prompted.

SFT bridges the gap between a powerful but "raw" autocomplete engine and a usable instruction-following assistant.

Key characteristics:
- **Supervised**: Uses explicit (prompt → response) pairs with cross-entropy loss on the desired tokens.
- **Fine-tuning**: Typically performed on the full model or with parameter-efficient methods ([LoRA](../PEFT/LoRA.md), [QLoRA](../PEFT/QLoRA.md), adapters).
- **Dataset size**: Usually 10k–200k high-quality examples (much smaller than pre-training data).
- **Goal**: Teach style, format, safety, helpfulness, and task-solving ability.

## Position in the Full Training Pipeline

| Stage              | Objective                          | Data Type                          | Typical Scale                  | Loss Function                  |
|--------------------|------------------------------------|------------------------------------|--------------------------------|--------------------------------|
| Pre-training       | Next-token prediction              | Unlabeled web text, books, code    | Trillions of tokens            | Causal language modeling       |
| **SFT**            | Instruction following              | Curated (prompt, response) pairs   | 10k–200k examples              | Causal LM on response tokens   |
| Preference Tuning (RLHF / DPO / KTO) | Alignment with human values       | Preference pairs or rankings       | 100k–1M preferences            | Reward modeling or direct pref |
| Post-training (safety, tools) | Additional capabilities & safeguards | Specialized datasets              | Varies                         | Varies                         |

SFT is the **foundation** for all later alignment stages. Poor SFT → poor alignment performance, even with excellent RLHF.

## Data Preparation

Quality > quantity. Common sources:

1. **Human-written datasets**
   - ShareGPT / Vicuna conversations
   - OpenAssistant
   - Dolly 15k
   - Alpaca (GPT-3.5/4 generated)

2. **Synthetic datasets**
   - Self-Instruct (Wang et al., 2022)
   - Evolution-based methods (e.g., UltraChat, Orca)
   - Distilled from stronger models (Phi-3, Llama-3 uses heavy synthetic data)

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
- **LoRA / QLoRA**: Low-rank adapters (r=16–64). QLoRA adds 4-bit quantization → fits 70B on single 48GB GPU.
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

- **Llama-2**: SFT on ~1M human + synthetic instructions → major capability jump over Llama-1.
- **Llama-3**: Heavy use of synthetic high-quality data + rejection sampling → state-of-the-art open 70B.
- **Grok-1 / Grok-2**: xAI’s models also undergo extensive SFT before alignment.
- **Mistral / Mixtral**: Fine-tuned versions (e.g., Mistral-7B-Instruct) dominate leaderboards with efficient SFT + minimal RLHF.