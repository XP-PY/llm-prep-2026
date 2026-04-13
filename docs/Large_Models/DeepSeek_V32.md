# [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)

**DeepSeek-V3.2** aims to **close the gap between open and frontier closed LLMs** by improving **(a) long-context efficiency**, **(b) post-training RL scalability**, and **(c) agent/tool-use generalization**—without sacrificing performance. 

<!-- When we go deeper, the most important low-level questions will be:

What exactly is stored in the KV cache under MLA-MQA?

What vectors does the lightning indexer use for $q^I$, $k^I$, and why those?

How do they implement Top-k selection and gathering efficiently at 128K?

How does this interact with long-context prefill vs decoding? -->

---

## 1) High-level contributions (the “3 pillars”)

### 1. DeepSeek Sparse Attention (DSA)

A new attention mechanism that **keeps long-context performance** while **reducing attention compute** in long sequences. 

### 2. Scalable RL framework (post-training compute scaled hard)

They emphasize that open models often under-invest in post-training. V3.2 claims a **post-training compute budget >10% of pretraining cost**, enabling reasoning performance comparable to GPT-5 (and a “Speciale” variant that pushes further). 

### 3. Large-scale agentic task synthesis pipeline

A pipeline that generates **many tool-use environments + prompts**, then uses RL to train for **generalizable agent behavior** (not just overfitting to a narrow toolset). 

---

## 2) Architecture changes vs DeepSeek-V3.1 / V3 (what actually changes)

The paper states: relative to **DeepSeek-V3.1-Terminus**, the **only architectural modification** in V3.2 is **introducing DSA** via continued training. 

So you can think of V3.2 as:

* V3.1-Terminus base (128K)
  * swap dense attention for DSA (sparse selection)
  * heavy post-training RL + agent synthesis

---

## 3) DeepSeek Sparse Attention (DSA)

### 3.1 Core idea

Instead of attending to **all previous tokens** $O(L^2)$, each query token attends to **top-k selected tokens** $O(L·k)$, where $k ≪ L$. 

<div align="center">
  <img src="../Resource/pics/DeepSeek-VL32_Attention_Architecture.png" alt="Attention architecture of DeepSeek-V3.2">
  <p style="text-align: center; font-size: smaller;">
    <strong>Fig 1: Attention architecture of DeepSeek-V3.2.</strong>
  </p>
</div>

### 3.2 Two-component design (Prototype)

**(A) Lightning Indexer**
Computes an **index score** $I_{t,s}$ between query token $h_t$ at position $t$ and each previous token $h_s$ at position $s$ to decide which tokens matter. Interpretation: “How likely is token $h_s$ to matter for token $h_t$?”

They define:
$$
I_{t,s}=\sum_{j=1}^{H_I} w^I_{t,j}\cdot \mathrm{ReLU}(q^I_{t,j}\cdot k^I_s)
$$

* $H_I$: number of indexer heads (small)
* $q^I_{t,j}$, $w^I_{t,j}$: derived from query token $h_t$
* $k^I_s$: derived from token $h_s$ shared across indexer heads
* ReLU chosen for throughput (Negative similarities contribute 0; Keeps computation simple and avoids exponentials here)
* Indexer can run efficiently (they mention FP8 feasibility). 

**(B) Fine-grained token selection**
For each query token $t$, pick the **top-k** tokens $s$ by index score $I_{t,s}$, retrieve only those KV entries, then run normal attention over that subset:
$$
u_t=\mathrm{Attn} (h_t, \{c_s \mid I_{t,s}\in \mathrm{Top}\text{-}k(I_{t,:})\} )
$$


### 3.3 How DSA is implemented inside MLA

They **instantiate DSA under MLA** (their [Multi-head Latent Attention](../Attention_Machanisms/MLA.md) setup from [earlier DeepSeek work](./DeepSeek_V2.md)) but specifically in a way that is kernel-efficient: they use **[MQA mode](../Attention_Machanisms/MQA.md)** so that each latent KV is shared across query heads. 

* The paper highlights a kernel-level constraint: **KV entries retrieved for sparse attention should be shareable across multiple query heads** for efficiency. so they choose MQA-mode MLA. 
  > If KV were per-head (classic full MHA), sparse retrieval would mean:
  > * either retrieving different KV sets per head (expensive / irregular)
  > * or doing awkward merging that kills throughput
* See Fig 1 for the architectural flow: “Lightning Indexer → Top-k Selector → Core attention uses only selected KV.”. 
* Fig 2 clarifies MLA’s **MHA vs MQA** modes. 

<div align="center">
  <img src="../Resource/pics/DeepSeek-VL32_MHA-and-MQA-modes-of-MLA.png" alt="Illustration of the MHA and MQA modes of MLA">
  <p style="text-align: center; font-size: smaller;">
    <strong>Fig 2: Illustration of the MHA and MQA modes of MLA.</strong>
  </p>
</div>

---

## 4) How they *train* DSA (continued pretraining)

They start from **DeepSeek-V3.1-Terminus** whose context length is already **128K**, then do continued pretraining in **two stages** with the same long-context data distribution. 

### 4.1 Stage 1: Dense warm-up (initialize the indexer)

* Keep **dense attention**
* **Freeze all parameters** except the lightning indexer
* Goal: make indexer outputs approximate the main attention distribution

**How they build the target distribution**

* For each query token $t$, aggregate main attention scores by summing over heads
* L1-normalize across sequence positions → target $p_{t,:}$

**Loss**
$$
L_I = \sum_t D_{KL} (p_{t,:} || \mathrm{Softmax}(I_{t,:}) )
$$

**Training settings:**
* LR: $10^{-3}$
* 1000 steps
* Each step: 16 sequences × 128K
* Total warm-up tokens: 2.1B 

### 4.2 Stage 2: Sparse training (turn on top-k selection)

* Enable token selection + train all model params to adapt to sparsity
* Still aligns indexer, but only on selected set $S_t={s \mid I_{t,s}\in\mathrm{Top}\text{-}k}$

**Loss becomes:**
$$
L_I = \sum_t D_{KL} (p_{t,S_t} || \mathrm{Softmax}(I_{t,S_t}) )
$$
Key training detail:

* They **detach the indexer input**: indexer learns only from $L_I$, main model learns only from LM loss. 

**Training settings:**

* LR: $7.3\times 10^{-6}$
* k = **2048 selected KV tokens** per query token
* 15000 steps
* Each step: 480 sequences × 128K
* Total tokens: 943.7B 

---

## 5) Efficiency & inference cost claims

### 5.1 Complexity

* Core attention reduced from **$O(L^2)$** to **$O(L·k)$** for $k≪L$. 
* Lightning indexer is still $O(L^2)$, but much cheaper than full attention and can run efficiently:
  * fewer heads ($H_I$ small)
  * simpler ops (ReLU, no full softmax over all heads for value mixing)
  * and they mention it can run efficiently (including FP8 feasibility)

### 5.2 Real service cost curves

Fig 3 shows **cost per million tokens vs token position** on H800 clusters (assumed cost: $2/GPU-hour). DeepSeek-V3.2 is cheaper than V3.1-Terminus for long-context prefilling and decoding. 

<div align="center">
  <img src="../Resource/pics/DeepSeek-VL32_Inference-cost.png" alt="Inference costs">
  <p style="text-align: center; font-size: smaller;">
    <strong>Fig 3: Inference costs.</strong>
  </p>
</div>

They also mention:

* For short-sequence prefilling they implement a “masked MHA mode” to simulate DSA for better short-context efficiency. 

---

## 6) Post-training pipeline (the main flow after continued pretraining)

**One-line picture:**

1. start from the DSA-upgraded base model,
2. train multiple **specialist models** with heavy RL,
3. use those specialists to generate domain data and **distill** a stronger general model,
4. run one **mixed RL stage** over reasoning + agent + alignment tasks,
5. choose either the more efficient official V3.2 recipe or the more reasoning-heavy **Speciale** recipe.

The paper says V3.2 keeps the same post-training recipe as **V3.2-Exp**. The key design choice is:

> instead of doing many separate RL stages one after another, they try to merge reasoning, agent, and alignment into a single large mixed-RL stage.

That is different in flavor from DeepSeek-R1, where the paper emphasizes a clearer multi-stage SFT/RL pipeline.

### 6.1 Why specialists appear first

They first train several **specialist models** from the same V3.2 base checkpoint using large-scale RL compute.

Domains include:

* math
* programming
* general logical reasoning
* general agent tasks
* agentic coding
* agentic search

Each specialist can have both **thinking** and **non-thinking** modes.

The intuition is simple:

* a specialist can push much harder in one domain,
* then its high-quality trajectories / outputs can be reused as training data,
* and a later distilled model can inherit that domain knowledge without maintaining many separate deployed models.

### 6.2 What “specialist distillation” means here

After specialist RL:

* the specialists generate domain-specific data for the final model,
* a distilled general model is trained on that specialist-generated data,
* this distilled model is already strong, but still slightly below the specialists,
* then a final RL stage is used to close the gap.

So the specialist stage is not the final serving model. It is more like a **data-and-capability generator** for the eventual unified model.

### 6.3 The mixed RL stage

After distillation, they run **[GRPO](../Preference_Alignment/GRPO.md)** on a mixed task distribution instead of doing completely separate RL phases for each ability.

The paper's motivation is to reduce **catastrophic forgetting**. If reasoning RL, agent RL, and alignment RL are done too separately, one stage may damage abilities built in another stage.

So the training mix contains at least three broad buckets:

* reasoning tasks,
* agent / tool-use tasks,
* general alignment tasks.

### 6.4 Official V3.2 vs V3.2-Speciale

The two published variants mainly differ in how aggressively they optimize for long reasoning:

* **DeepSeek-V3.2 (official)**: keeps stronger token / efficiency constraints, aiming for a better quality-cost balance.
* **DeepSeek-V3.2-Speciale**: focuses more heavily on reasoning, trains only on reasoning data, reduces the length penalty, and adds DeepSeekMath-V2 data and reward design to strengthen proof-style ability.

So:

* **official V3.2** = more balanced production-oriented model
* **Speciale** = more reasoning-maximized variant

---

## 7) What data and rewards feed the post-training pipeline?

DeepSeek-V3.2 is not trained on one uniform post-training dataset. Instead, it combines several task/data/reward regimes.

### 7.1 Reasoning and agent tasks: mostly rule-based rewards

For reasoning and agentic tasks, the paper emphasizes rewards that can be checked automatically.

Examples mentioned:

* **rule-based outcome reward**
* **length penalty**
* **language consistency reward**

This is the same general philosophy in R1 and GRPO:

* if the environment can tell you whether the answer is correct,
* RL becomes much more scalable and much less dependent on fragile learned reward models.

### 7.2 General assistant tasks: generative reward model + rubric

For more subjective tasks, V3.2 does not rely only on rule-based verification.

Instead, it uses:

* a **generative reward model**
* **per-prompt rubrics**

This means the reward signal is more like:

* "Given this prompt, what does a good answer look like?"
* "Score the answer according to that rubric."

So the post-training setup is mixed:

* **objective / verifiable tasks** -> rule-based rewards
* **subjective / alignment tasks** -> rubric-guided model-based rewards

### 7.3 Why this mixed reward design matters

This is one of the main engineering lessons from the paper:

* do not force everything into one reward style,
* use **verification** where the environment is reliable,
* use **rubric-based judging** where the task is subjective.

That hybrid design is part of why the pipeline can cover both reasoning and general assistant behavior.

---

## 8) Scaling GRPO in practice (why RL does not collapse)

Once you understand the main pipeline, the next question is: how do they make GRPO stable at this scale?

They restate the GRPO objective in the paper and then add several stabilizers.

### 8.1 Unbiased KL estimate

They adjust KL estimation with importance sampling to make gradients less biased, arguing that the original estimator can produce bad gradients when $ \pi_\theta \ll \pi_{ref} $.

You can read this as:

* the KL term is important for trust-region behavior,
* but if its estimator is numerically poor, RL can become unstable.

### 8.2 Off-policy sequence masking

They reuse rollout data across multiple updates, so the data can drift off-policy. They also note implementation differences between sampling and training.

To reduce the damage from that drift, they:

* measure divergence between the sampling policy $ \pi_{old} $ and current policy $ \pi_\theta $,
* mask sequences with **negative advantage** and **too large divergence**.

So not every collected rollout is trusted equally during training.

### 8.3 Keep Routing

Because the model is MoE, routing paths may differ between sampling-time inference and training-time recomputation.

They therefore:

* record routing paths during sampling,
* replay / enforce them during training.

This is a very practical detail. Without it, the "same" sampled trajectory is no longer really the same action under the training graph.

### 8.4 Keep Sampling Mask

Top-p / top-k truncation changes the effective action space. That breaks the assumptions behind importance weighting if training later recomputes probabilities over a different candidate set.

So they preserve the original truncation mask from $ \pi_{old} $ and reuse it during training, so current and old policy are compared over the same action subset.

---

## 9) Agent training data engine (large-scale task synthesis)

A major claim of V3.2 is that open models were under-investing not only in RL compute, but also in **agent environments and task synthesis**.

The core idea is:

> build environments where solving the task is hard, but verifying success is easy.

That gives RL a scalable reward signal.

The paper reports four major task buckets:

* **Code agent**: 24,667 tasks, real environments, extracted prompts
* **Search agent**: 50,275 tasks, real environments, synthesized prompts
* **General agent**: 4,417 tasks, synthetic environments, synthetic prompts
* **Code interpreter**: 5,908 tasks, real environments, extracted prompts

### 9.1 Search agent: synthetic questions in real search environments

The search pipeline is explicitly multi-agent:

1. sample long-tail entities from web corpora,
2. let a question-construction agent explore those entities with search tools,
3. generate multiple candidate answers,
4. verify them with a search-based verifier,
5. keep only samples whose correctness is verifiable.

So the important output is not just "a question". It is a question paired with a search environment and a reliable way to check whether an answer is supported.

### 9.2 Code agent: executable issue-resolution tasks

For coding agents, they mine GitHub issue-PR pairs and try to turn them into executable environments.

An environment is kept only if:

* the gold patch turns some failing tests into passing ones (`F2P > 0`),
* and does not break previously passing tests (`P2F = 0`).

That means the reward is grounded in actual execution, not just in a human preference score saying "this patch looks good."

### 9.3 General agent: synthetic environments with built-in verifiers

For general agents, they synthesize:

* a small sandbox database,
* task-specific tools / functions,
* tasks,
* a solution function,
* and a verification function.

Then they iteratively raise difficulty and keep only tasks that are neither trivial nor impossible.

The trip-planning example in the paper is a good mental model:

* many constraints,
* several tools,
* structured output,
* and a programmatic verifier.

### 9.4 Code interpreter: execution-heavy problems in notebook-like environments

The fourth bucket is **code interpreter** training.

The idea is:

* give the model an execution environment,
* ask it to solve problems that require writing and running code,
* and score it based on whether the execution actually produces the right result.

The paper describes these as real environments with extracted prompts, covering problems from areas like:

* math,
* logic,
* data analysis / data science.

So this bucket sits between "reasoning" and "agent use":

* it is not just static QA,
* but it is also not full open-ended web/search agency,
* and execution gives a clean verifier.

### 9.5 Why this data engine matters

The key point is not just that there are many tasks. It is that each task comes with:

* an **environment**,
* a **prompt**,
* and a **verifier / reward path**.

That is what makes agentic RL possible at scale. Without the environment and verifier, you would only have static QA data, not true agent-training data.

---

## 10) “Thinking in Tool-Use” (how reasoning and tools are combined)

One subtle point in the paper is that they do not treat "reasoning model" and "tool-use model" as completely separate things.

### 10.1 Thinking context management

They keep reasoning traces **across tool calls**, and only discard them when a **new user message** arrives.

Tool call history and tool results remain preserved even if some reasoning trace is later removed.

This matters because many agent tasks are not "think once, then act once." They are:

* think,
* call tool,
* observe result,
* think again,
* call another tool,
* then answer.

The paper also warns that some frameworks convert tool interactions into new user messages. That weakens this benefit, so they recommend non-thinking models in such frameworks.

### 10.2 Cold-start for tool-using reasoning

They seed the model with a mixture of:

* reasoning data using `<think>...</think>`,
* non-reasoning agent data in tool-call format,
* a system prompt that demonstrates "tool calls inside thinking."

The purpose is not to fully solve agent behavior with supervised data. It is to create enough initial successful trajectories that later RL can expand on them.

---

## 11) Evaluation overview (what they measure and how)

### 11.1 Benchmarks (selected)

They evaluate on a large set including:

* reasoning/knowledge: MMLU-Pro, GPQA, HLE (text-only)
* math: AIME 2025, HMMT 2025, IMOAnswerBench
* coding: LiveCodeBench, Codeforces
* agent/tool: Terminal Bench 2.0, SWE-Verified, BrowseComp (+Zh), τ²-bench, MCP-Universe, MCP-Mark, Tool-Decathlon 

### 11.2 Tool-use eval settings

* tool-use benchmarks use function call format with thinking mode
* temperature 1.0
* context window 128K 

### 11.3 Main result narrative

* DeepSeek-V3.2: ~GPT-5-High level on many reasoning tasks, below Gemini-3.0-Pro overall
* Strong gains in coding-agent benchmarks vs open models
* Tool-use still below frontier but gap narrowed 

### 11.4 Speciale results

Speciale improves accuracy by using more reasoning tokens, and reports “gold-level” performance on competitions (IOI/IMO/ICPC/CMO tables). 
But token efficiency is worse; official V3.2 uses stricter length constraints to balance cost. 

---

## 12) Context management for search agents (test-time compute scaling)

When token usage > 80% of context length, they apply strategies:

* **Summary** (summarize overflowed trajectory then re-rollout)
* **Discard-75%** tool history
* **Discard-all** tool history (like “new context tool”)
  They compare to a parallel baseline (sample N trajectories). Discard-all performs well, reaching 67.6 on BrowseComp in their setting. 

---

## 13) Limitations & future work (their own admission)

They cite three main limitations vs frontier closed models:

1. **World knowledge breadth** lags due to fewer total training FLOPs → plan to scale pretraining compute
2. **Token efficiency**: needs longer reasoning trajectories → improve “intelligence density”
3. Still inferior on hardest tasks → refine foundation + post-training recipe 

---

## 14) Mental model: “What changed from V3 to V3.2?”

If you already learned V2/V3, a useful compression is:

* **Pretrain/Backbone**: basically same family
* **Key arch delta**: **DSA** (sparse selection via a learned indexer) to make 128K cheaper
* **Key capability delta**: **scale RL** + **agent synthesis** to push reasoning + tool-use generalization
* **Speciale**: “remove some efficiency constraints and go all-in on reasoning tokens”

All supported by the paper’s description and figures/tables. 
