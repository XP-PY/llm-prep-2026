# [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

## Convenient Links
* [Github](https://github.com/deepseek-ai/DeepSeek-R1)
* Hugging Face:
    * [DeepSeek-R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)
    * [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
* [Nature paper (published September 17, 2025)](https://www.nature.com/articles/s41586-025-09422-z)

## One-line summary

**DeepSeek-R1** is a reasoning-specialized model family built on **DeepSeek-V3-Base** that made a strong case for **large-scale RL as the main driver of reasoning ability**, rather than relying only on supervised chain-of-thought demonstrations.

---

## 1. What DeepSeek-R1 is

The family has two main large models:

* **DeepSeek-R1-Zero**: pure RL on the base model, without SFT before RL.
* **DeepSeek-R1**: a refined version that adds **cold-start data + multi-stage SFT/RL** to fix readability and alignment problems.

Shared core specs:

* **671B total parameters**
* **37B activated parameters**
* **128K context length**
* based on **[DeepSeek-V3-Base](./DeepSeek_V3.md)** architecture

The official release happened on **January 22, 2025**, and the peer-reviewed **Nature** version appeared later on **September 17, 2025**.

---

## 2. Why DeepSeek-R1 matters

Before R1, the common intuition was:

* pretrain a strong base model,
* collect many supervised reasoning traces,
* then do RLHF or other preference optimization.

DeepSeek-R1 challenged this by showing that:

* **reasoning behaviors can emerge from RL itself**,
* especially on tasks with a **reliable verifier**,
* and long chain-of-thought can be **discovered**, not only imitated.

The paper explicitly reports emergent behaviors such as:

* **self-reflection**
* **verification**
* **trying alternative solution paths**
* an "aha moment" during RL where reasoning patterns visibly shift

This is why R1 became a landmark model in the "reasoning via RL" direction.

---

## 3. DeepSeek-R1-Zero: pure RL first

### 3.1 Core idea

DeepSeek-R1-Zero applies **[GRPO](../Preference_Alignment/GRPO.md)** directly to the base model without an SFT warm-start.

The official Nature paper says the reward is based only on:

* **accuracy reward**
* **format reward**

For reasoning tasks, they intentionally avoid neural reward models because those are more vulnerable to **reward hacking** during large-scale RL.
```
reward hacking: The model discovers patterns that increase the reward score but do not correspond to correct behavior
```

### 3.2 What the rewards look like

For verifiable tasks:

* **math**: check whether the final answer is correct
* **coding / code competition**: compile and run against test cases
* **format**: enforce explicit reasoning tags such as `<think>...</think>`

So the training signal is simple in spirit:

* solve the problem correctly
* show the answer in a machine-checkable format

### 3.3 What emerged

Without being directly taught a fixed reasoning script, R1-Zero learned to:

* spend more tokens on hard problems,
* revisit earlier steps,
* self-check answers,
* and explore alternatives before finalizing a solution.

This is the strongest conceptual point of the paper.

### 3.4 Why R1-Zero was not enough

Although R1-Zero reasoned well, it had several practical issues:

* **poor readability**
* **language mixing** (especially English/Chinese mixture in CoT)
* weak performance on broader general-assistant tasks like writing and open-domain QA

So DeepSeek used R1-Zero as a proof-of-concept, not as the final user-facing model.

---

## 4. DeepSeek-R1: the four-stage pipeline

DeepSeek-R1 keeps the same base architecture, but changes the **post-training recipe**.

### Stage 1: cold-start SFT

They collect **thousands of cold-start samples** with a more conversational, human-aligned reasoning style.

Goal:

* improve readability,
* improve instruction following,
* reduce messy raw reasoning behavior from pure RL.

### Stage 2: reasoning-oriented RL

They then run RL again on reasoning data.

Important detail:

* they add a **language consistency reward** to reduce language mixing in CoT.

This stage mainly boosts:

* math
* code generation
* STEM reasoning

### Stage 3: rejection sampling + SFT again

After RL, they use **rejection sampling** and another SFT stage.

This time the SFT mix includes:

* **reasoning data**
* **non-reasoning data**

The point is to keep the reasoning gains while restoring stronger general writing and assistant behavior.

### Stage 4: second RL stage for broad alignment

Finally, they run another RL stage to improve:

* **helpfulness**
* **harmlessness**
* while still preserving reasoning ability

So the final DeepSeek-R1 is not "pure RL only". The key message is:

> **R1-Zero proves pure RL can discover reasoning; R1 turns that into a more usable assistant via multi-stage SFT + RL.**

---

## 5. Reward design: why R1 works best on verifiable tasks

This is one of the most important ideas to remember.

For **reasoning-oriented data**, DeepSeek-R1 still relies on rule-based rewards:

* correctness
* format
* language consistency

For **general assistant data**, it uses model-based rewards:

* a **helpfulness reward model**
* a **safety reward model**

This split is important:

* if the task is easy to verify, RL works very well
* if the task is subjective and hard to verify, reward modeling is still needed

So R1 is not claiming that all alignment can be solved by pure RL. It is strongest where the environment can say "correct" or "incorrect" reliably.

---

## 6. Distillation: a second major contribution

DeepSeek did not only release the big 671B MoE models. They also distilled R1 into smaller dense models based on:

* **Qwen2.5**
* **Llama 3.x**

Open-sourced distilled sizes include:

* 1.5B
* 7B
* 8B
* 14B
* 32B
* 70B

The important lesson from the paper:

* **distilling strong reasoning traces from a large RL-trained model can outperform trying to do RL directly on much smaller models.**

This is why the distilled R1 models became widely used in practice.

---

## 7. Selected benchmark signals

The official model card reports the following numbers for **DeepSeek-R1**:

* **AIME 2024 (Pass@1): 79.8**
* **MATH-500 (Pass@1): 97.3**
* **GPQA-Diamond (Pass@1): 71.5**
* **LiveCodeBench (Pass@1-COT): 65.9**
* **Codeforces rating: 2029**
* **SWE-bench Verified (Resolved): 49.2**
* **MMLU (Pass@1): 90.8**

Interpretation:

* extremely strong on **math** and **competitive reasoning**
* strong on **coding**
* more mixed on broad factual QA or software-engineering tasks than the headline reasoning results may suggest

---

## 8. Practical limitations

The paper and model card emphasize several caveats:

* **tool use is not built in** for R1 itself
* **structured output** is weaker than some assistant-oriented models
* the model is **prompt-sensitive**
* **few-shot prompting can hurt performance**
* optimization is strongest for **Chinese and English**
* large-scale RL was **not applied extensively to software-engineering tasks**, so gains there are more limited than in math/code competitions

The model card also recommends:

* temperature around **0.5-0.7** (with **0.6** recommended)
* avoid unnecessary system prompts
* for math, explicitly ask for step-by-step reasoning and a boxed final answer

---

## 9. What to remember

If you only remember three things about DeepSeek-R1, remember these:

1. **R1-Zero** showed that large-scale RL can make reasoning emerge without SFT reasoning traces.
2. **R1** itself is a **hybrid multi-stage pipeline**, not pure RL only.
3. The method works best when tasks are **verifiable**, so reward signals can stay reliable.
