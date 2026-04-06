# Direct Preference Optimization (DPO) for LLM Alignment

## Overview
Direct Preference Optimization (DPO) aligns a language model **directly on preference pairs**
\((x, y^+, y^-)\) **without** training a separate reward model and **without**
running an on-policy RL loop such as [PPO](./PPO.md). It turns
"make the chosen response more likely than the rejected one" into a stable supervised objective,
while still keeping the model close to a frozen reference policy through an **implicit KL regularization** effect.

**One-line intuition:**  
> DPO trains the model so that (chosen − rejected) gets a positive *log-odds margin* compared to a frozen reference model.

---

## 1) Where DPO sits in the pipeline
| Stage | Classic RLHF | DPO-style |
|:---:|:---:|:---:|
| Base pretrain | ✓ | ✓ |
| SFT / instruction tuning | usually ✓ (strongly recommended) | usually ✓ (strongly recommended) |
| Preference data | (x, y⁺, y⁻) | same |
| Reward model | train RM | **skip** |
| Policy optimization | PPO (on-policy) + explicit KL | **offline** preference optimization (supervised) |
| Stability / engineering | harder | easier |

DPO is most effective when the **reference policy is already good** (often the SFT checkpoint).

---

## 2) The key object: reference-corrected preference score
Define:

* $\pi_\theta$: your trainable model
* $\pi_{\text{ref}}$: a frozen reference model, usually the SFT checkpoint

For a prompt $x$ (input text) and completion $y$ (the model’s answer text), define the **reference-corrected score**:
$$
s_\theta(x,y) := \log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)
$$
This measures:

> How much more or less does my current model like this answer compared with the reference model?

Here $\pi_\theta(y|x)$ is the probability of generating the **whole completion** $y$ given prompt $x$.
Because $y$ is a token sequence \(y_1, y_2, \dots, y_T\), in practice:
$$
\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t})
$$

For a preference pair ($y^+$ preferred over $y^-$), the margin is:
$$
\Delta s_\theta := s_\theta(x,y^+) - s_\theta(x,y^-)
$$
where $y^+$ is the chosen response and $y^-$ is the rejected response. Expand it:
$$
\Delta s_\theta = \big( \log \pi_\theta(y^+|x) - \log \pi_{\text{ref}}(y^+|x) \big) - \big( \log \pi_\theta(y^-|x) - \log \pi_{\text{ref}}(y^-|x) \big)
$$

Regroup:
$$
\Delta s_\theta = 
\underbrace{
\left( \log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x) \right)
}_{\text{how much }\theta\text{ prefers chosen over rejected}} - 
\underbrace{
\left( \log \pi_{\text{ref}}(y^+ \mid x) - \log \pi_{\text{ref}}(y^- \mid x) \right)
}_{\text{how much ref prefers chosen over rejected}}
$$

$\Delta s_\theta$ literally means:
- Compared with the reference model, how much more does my current model favor the chosen answer over the rejected answer?

So:
- If $\Delta s_\theta > 0$: good
- If $\Delta s_\theta < 0$: bad
- If $\Delta s_\theta = 0$: your model is no better than reference on this pair

---

## 3) DPO loss (what is actually implemented)
DPO models the probability that $y^+$ is preferred over $y^-$ as:
$$
P(y^+ \succ y^-|x) = \sigma(\beta \, \Delta s_\theta)
$$
and maximizes the log-likelihood. So the loss is:
$$
\mathcal{L}_{\text{DPO}}(\theta) =
-\mathbb{E}_{(x,y^+,y^-)} \left[ \log \sigma(\beta \, \Delta s_\theta) \right]
$$

### What β does (important)
- β controls **how aggressively** you move away from $\pi_{\text{ref}}$.
- Larger β ⇒ stronger push to separate chosen from rejected (can “over-align”, become bland / brittle).
- Smaller β ⇒ stays closer to reference (may under-align).

In practice β is task/model dependent; common starting points are around **0.05–0.5**.

---

## 4) Where this comes from (derivation sketch, minimal but correct)
Start from the KL-regularized RL objective (the “RLHF view”):
$$
\max_\pi \; \mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)] - \beta \, D_{KL}(\pi(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))
$$
This has a closed-form optimal policy:
$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)
$$
so reward differences correspond to **log-ratio differences**:
$$
r(x,y_1)-r(x,y_2) = \beta\left[\log\frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}-\log\frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}\right]
$$
Assume preferences follow Bradley–Terry:
$$
P(y^+\succ y^-|x)=\sigma(r(x,y^+)-r(x,y^-))
$$
Replace the unknown reward difference with the policy log-ratio difference, and optimize θ to
match those preferences. This yields the DPO objective in Section 3.

---

## 5) Practical details that matter

### 5.1 Computing log π(y|x)
In code, \(\log \pi_\theta(y|x)\) is the **sum of token logprobs of the completion tokens**,
conditioned on the prompt. (Mask prompt tokens; only score completion tokens.)

### 5.2 Reference model choice
- Usually $\pi_{\text{ref}}$ is the **SFT checkpoint** (frozen).
- Some setups use the same architecture/weights snapshot as reference.
- If reference is weak, DPO can drift or learn weird shortcuts.

### 5.3 Data format (minimum)
Each example needs:
- `prompt`
- `chosen` (preferred response)
- `rejected` (dispreferred response)

Quality beats quantity: noisy preference pairs can harm more than help.

---

## 6) Common failure modes (and how to notice them)
1) **Verbosity / length bias**
- Symptom: model gets overly long or overly short.
- Mitigation: balanced preference data; length normalization variants (e.g., SimPO-style); evaluation with length-controlled prompts.

2) **Overfitting to preference style**
- Symptom: “sounds aligned” but worsens factuality/reasoning.
- Mitigation: diverse domains; holdout eval; mix some SFT data (or do multi-objective training).

3) **Too strong β**
- Symptom: bland, refusal-heavy, loses creativity/helpfulness.
- Mitigation: sweep β downward; monitor win-rate + qualitative samples.

4) **Bad pairs**
- Symptom: instability, regressions, reward hacking-like behavior.
- Mitigation: filter ties/low-confidence; dedup; remove contradictory labels.

---

## 7) Key variants (high-level)
| Method | Idea | Why it exists |
|:---:|:---:|:---:|
| DPO | logistic on log-ratio margin | simple baseline |
| IPO | squared loss variant | smoother gradients in some regimes |
| SimPO | length-aware / reward shaping | reduce verbosity bias |
| ORPO | mix SFT + preference penalty | “one-stage” training feel |
| KTO | works with (good/bad) signals | weaker feedback settings |

---

## 8) Minimal TRL usage (conceptual)
- Prepare dataset with `prompt/chosen/rejected`
- Load model + reference model (frozen)
- Run DPOTrainer with a reasonable β, lr, batch, and LoRA (optional)
- Evaluate win-rate vs SFT baseline on heldout prompts

## 9) Practical Implementation Example with Hugging Face TRL
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# --------------------------------------------------
# 1. Load model and tokenizer
# --------------------------------------------------
model_name = "Qwen/Qwen2.5-7B-Instruct"   # replace with your SFT checkpoint if available

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# Reference model is usually the frozen SFT model
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# --------------------------------------------------
# 2. Load preference dataset
# --------------------------------------------------
dataset = load_dataset("Intel/orca_dpo_pairs")

train_dataset = dataset["train"]

# Optional: inspect one example
print(train_dataset[0])

# --------------------------------------------------
# 3. Optional LoRA config
# --------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# --------------------------------------------------
# 4. DPO training config
# --------------------------------------------------
training_args = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                          # key DPO hyperparameter
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    bf16=True,                         # use fp16=True if bf16 is not supported
    max_length=1024,
    max_prompt_length=512,
    remove_unused_columns=False,
    report_to="none",
)

# --------------------------------------------------
# 5. Create trainer
# --------------------------------------------------
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# --------------------------------------------------
# 6. Train
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# 7. Save model
# --------------------------------------------------
trainer.save_model("./dpo_output/final_checkpoint")
tokenizer.save_pretrained("./dpo_output/final_checkpoint")
```
