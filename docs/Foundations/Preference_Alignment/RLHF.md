# Reinforcement Learning from Human Feedback (RLHF) in Large Language Models

## Overview

Reinforcement Learning from Human Feedback (RLHF) is the key **alignment** stage that turns an [instruction-tuned (SFT)](./SFT.md) model into a helpful, honest, and harmless assistant. While SFT teaches the model to follow instructions, RLHF teaches it to produce responses that **humans actually prefer**—prioritizing helpfulness, safety, coherence, and style.

Classic RLHF (InstructGPT / ChatGPT style) has three phases:
1. **Supervised Fine-Tuning (SFT)** → produces reasonable responses.
2. **Reward Model (RM) Training** → learns a scalar reward function from human preference data.
3. **Policy Optimization** → fine-tunes the SFT model via reinforcement learning (usually **[PPO, Proximal Policy Optimization](./PPO.md)**) to maximize the learned reward while staying close to the original model.

RLHF is computationally expensive and can be unstable, which is one reason modern open-source pipelines often prefer **[direct preference optimization methods (DPO, KTO, ORPO)](./DPO.md)** that skip the explicit reward model and RL loop.

## What RLHF Adds Beyond SFT

Compared with SFT, classic RLHF adds three new ingredients:

1. **Preference data**: humans compare multiple candidate answers and indicate which one is better.
2. **Reward model**: a learned scoring model that predicts which answers are preferred.
3. **Policy optimization**: reinforcement learning updates that push the model toward higher-reward behavior while controlling drift.

So SFT mainly answers:

> Can the model follow the instruction and produce a reasonable answer?

RLHF goes one step further and answers:

> Among several reasonable answers, which one is more preferred?

## Position in the Full Training Pipeline

| Stage                  | Objective                              | Data Type                              | Typical Scale                     | Key Techniques                     |
|------------------------|----------------------------------------|----------------------------------------|-----------------------------------|------------------------------------|
| Pre-training           | Next-token prediction                  | Unlabeled text/code                    | Trillions of tokens               | Causal LM                          |
| SFT                    | Instruction following                  | High-quality (prompt, response) pairs  | 10k–200k examples                 | Supervised LM                      |
| **RLHF (Classic)**     | Preference alignment                   | Preference pairs (response A vs B)     | 10k–200k ranked pairs             | RM + PPO                           |
| **Direct Pref Opt**    | Preference alignment (no RM/RL)        | Preference pairs                       | Same as above                     | DPO / KTO / ORPO                   |
| Safety / Tools         | Additional safeguards & capabilities   | Specialized datasets                   | Varies                            | Red-teaming, tool-use training     |

RLHF and its alternatives are a major way to shape post-SFT model behavior, especially on helpfulness, harmlessness, tone, and response preference.

## Data Preparation for Preference Tuning

### Preference Data Collection
- Human annotators are shown a prompt and **two model responses** (A and B).
- They rank: A > B, B > A, or tie.
- Common sources:
  - OpenAI’s InstructGPT data (proprietary)
  - Anthropic’s HH-RLHF (helpfulness & harmlessness splits)
  - LMSYS Chatbot Arena (crowdsourced)
  - OpenAssistant, UltraFeedback, Nectar

### Dataset Examples
| Dataset          | Size (pairs) | Characteristics                          |
|------------------|--------------|------------------------------------------|
| HH-RLHF          | ~160k        | Helpfulness & Harmlessness separated     |
| UltraFeedback    | ~200k        | Multi-aspect ratings (helpfulness, honesty, etc.) |
| Nectar           | ~300k        | High-quality synthetic + human           |
| Argilla Pref Pairs| Varies      | Open-source, easy to extend              |

Best practices:
- Diverse prompts (open-ended, closed-ended, safety, reasoning)
- Balanced winning/losing responses
- Remove ties or low-confidence annotations
- Decontaminate against benchmark leakage

## Classic RLHF: Detailed Breakdown

### 1. Reward Model (RM) Training
- Initialize from the SFT model.
- Add a **linear head** on top of the last token to predict a scalar reward.
- Loss: Binary classification ranking loss (chosen > rejected)

```math
\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( r_\phi(y_w | x) - r_\phi(y_l | x) \right) \right]
```

where:

* $r_\phi$ is the reward model,
* $y_w$ is the chosen response,
* $y_l$ is the rejected response.

### 2. Policy Optimization with PPO
- Treat the LLM as the policy $\pi_\theta$.
- $\text{Reward} = \text{RM score} − \beta \times \text{KL}(\pi_\theta \parallel \pi_{\text{SFT}})$ (KL penalty prevents drift).
<!-- - Reward = RM score − β × KL(π_θ || π_SFT) (KL penalty prevents drift). -->
- Use [PPO (clipped surrogate objective)](./PPO.md) for stable updates.
- Often includes a **value head** for GAE advantage estimation.

Key hyperparameters (typical 7B–70B):
| Parameter              | Common Value           | Notes                                      |
|------------------------|------------------------|--------------------------------------------|
| KL coefficient β       | 0.01–0.05              | Controls drift from SFT                    |
| PPO epochs             | 1–4 per batch          |                                            |
| Mini-batch size        | 64–512                 |                                            |
| Clip range             | 0.2                    | Standard PPO clip                          |
| Learning rate          | 1e-6 – 5e-6            | Very small to avoid collapse               |
| LoRA rank (for PEFT)   | 64–256                 | Higher than SFT                            |

## Modern Alternatives (Strongly Recommended for Practice)

| Method | Key Idea                                  | Advantages                              | Disadvantages                     |
|:--------:|:-------------------------------------------:|:-----------------------------------------:|:-----------------------------------:|
| [DPO](./DPO.md)    | Directly optimizes preference log-loss without RM | Simpler, stable, better performance    | Needs strong SFT initial model    |
| KTO    | Uses Kahneman-Tversky objective (human-like)     | Works with binary feedback only        | Slightly more complex math        |
| ORPO   | Adds odds ratio penalty during SFT                | No separate preference data needed     | Early stage                        |

**DPO is currently the go-to preference-optimization method in many open-source pipelines** because it is simpler and more stable than classic PPO-based RLHF.

## Practical Open-Source Path (DPO Example with TRL)

The code example below is intentionally a **DPO** example rather than a classic PPO-based RLHF example. That matches common open-source practice today: teams often use SFT first, then prefer DPO-like methods for preference optimization because they are easier to run and debug.

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

model_name = "meta-llama/Llama-3-8b-hf"  # or your SFT checkpoint
dataset = load_dataset("Intel/orca_dpo_pairs")["train"]  # excellent open dataset

# DPO config
dpo_config = DPOConfig(
    beta=0.1,                    # temperature for preference strength
    output_dir="./llama3-8b-dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=5e-7,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,              # can provide explicit reference or use internal
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,     # LoRA for efficiency
)

trainer.train()
```

If you specifically want classic RLHF, use `PPOTrainer` (or an equivalent PPO pipeline) and expect substantially more complexity in rollout generation, reward scoring, advantage estimation, and training stability.

## Common Pitfalls & Mitigations

| Issue                     | Symptom                              | Fix                                           |
|---------------------------|--------------------------------------|-----------------------------------------------|
| Reward hacking            | Model exploits RM loopholes          | Diverse preference data, multi-aspect rewards |
| Mode collapse / repetition| Responses become bland or repetitive | Stronger KL penalty, length normalization     |
| Overfitting to RM         | Good on train prefs, poor generation | Early stopping, evaluation on held-out humans |
| Training instability      | Loss spikes, NaNs                    | Small LR, gradient clipping, good initialization |
| Length bias               | RM favors longer responses           | Length normalization in RM                    |

## Real-World Examples

- **InstructGPT / ChatGPT-style assistants**: proprietary RLHF-style pipelines built on large-scale preference data.
- **Llama-2-Chat**: RLHF with safety-specific rejection sampling.
- **Claude series**: Constitutional AI (similar spirit but rule-based reward).
- **Zephyr-7B**: SFT → DPO on UltraFeedback → beats many 70B models.
- **Llama-3 family**: large-scale post-training with instruction tuning, rejection sampling, and preference optimization.

## Future Directions

- **Iterative alignment** (RLAIF, self-critique loops)
- **Multimodal RLHF** (video, image preferences)
- **Scalable oversight** (AI-assisted labeling)
- **Direct Preference Optimization variants** dominating open-source (simpler & stronger)
