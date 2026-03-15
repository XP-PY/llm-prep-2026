# Group Relative Policy Optimization (GRPO) for LLM Reasoning

## 1) What GRPO is

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning method for large language models that was introduced in the **[DeepSeekMath](https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com)** work. It is designed to improve reasoning ability while using **less memory than PPO**, mainly by **removing the separate critic/value model** and estimating “advantage” from a **group of sampled answers for the same prompt**.

**One-line intuition:**

> For one prompt, sample several answers, score them, and make the model increase probability of the better-than-average answers while decreasing probability of the worse-than-average ones.

So GRPO is:

* **RL-style** post-training, not supervised fine-tuning.
* **On-policy**: it trains on answers sampled from the current model.
* **Critic-free**: it does not need a separate value network like PPO does.

---

## 2) Why GRPO was proposed

In PPO-style RLHF for LLMs, you usually need:

* a **policy model** (the LLM you update),
* a **reference model** for KL control,
* and often a **value model / critic** to estimate advantage.

That makes training heavier and more memory-hungry. GRPO was proposed to simplify this by:

1. sampling a **group** of responses for each prompt,
2. scoring them with a reward function,
3. computing each response’s quality **relative to the others in the same group**,
4. updating the model using those relative scores.

This is especially natural for tasks with **verifiable rewards**, such as:

* math
* code
* rule-based answer checking
* structured reasoning tasks

because you can score each output automatically.

---

## 3) The big picture: how GRPO works

For a prompt $x$:

1. Use the current policy $\pi_\theta$ to sample a **group** of completions:
   $$
   y_1, y_2, \dots, y_G
   $$
   where $G$ is the group size.

2. Compute a reward for each completion:
   $$
   r_1, r_2, \dots, r_G
   $$

3. Turn these raw rewards into **relative advantages** inside the group.

4. Increase probability of completions with **positive relative advantage**.

5. Decrease probability of completions with **negative relative advantage**.

6. Add a trust-region style constraint, usually via **clipping** and/or **KL control**, so updates do not become too aggressive. GRPO is described as a PPO variant, and TRL’s GRPO trainer also exposes KL-related controls in practice.

---

## 4) The key idea: “relative” reward inside a group

Suppose for the same prompt you sample 4 answers and get rewards:

$$
r = [1, 0, 1, 0]
$$

If you only looked at absolute rewards, you might say:

* reward 1 is good
* reward 0 is bad

But GRPO says something slightly more refined:

> What matters is whether this answer is better or worse than the other answers generated for the same prompt.

A common grouped advantage form is:

$$
A_i = \frac{r_i - \mu}{\sigma}
$$

where:

* $\mu$ = mean reward in the group
* $\sigma$ = standard deviation of rewards in the group

So GRPO normalizes reward **within the sampled group for that prompt**. Follow-up theory and analyses describe GRPO in terms of exactly this kind of within-group normalization.

Interpretation:

* $A_i > 0$: this answer is better than average for this prompt
* $A_i < 0$: this answer is worse than average
* $A_i = 0$: average

This is why it is called **Group Relative** Policy Optimization.

---

## 5) Why GRPO does not need a critic

In PPO, advantage is often estimated using a learned value model:
$$
A \approx R - V
$$

In GRPO, instead of learning a separate value function $V$, the algorithm uses the **other sampled responses in the same group** as a baseline.

So the group itself acts like a cheap, local estimate of:

* what is good for this prompt,
* what is bad for this prompt,
* and how strongly one answer stands out.

That is the main memory/computation-saving idea of GRPO.

---

## 6) The probability ratio part

Like PPO, GRPO still compares the **new policy** to the **old policy**.

For a sampled completion $y_i$, define the token-sequence probability ratio:

$$
\rho_i(\theta) =
\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}
$$

This says:

> Under the updated model, how much more likely is this sampled answer than it was under the model that generated it?

* If $\rho_i > 1$, the new model likes it more.
* If $\rho_i < 1$, the new model likes it less.

In practice, this is computed from **sequence log-probabilities**:
$$
\log \rho_i(\theta) = \log \pi_\theta(y_i|x) - \log \pi_{\theta_{\text{old}}}(y_i|x)
$$

---

## 7) The core GRPO objective

GRPO is commonly described as a PPO-style clipped objective, but with **group-relative normalized advantage** instead of critic-based advantage. A representative form is:

$$
\mathcal{L}_{\text{GRPO}}(\theta) = - \mathbb{E}\left[
  \frac{1}{G}
  \sum_{i=1}^{G}
  \min\Big(
  \rho_i(\theta) A_i,
  \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i
  \Big)
  \right]
$$

where:

* $G$ = group size
* $A_i$ = relative advantage inside the group
* $\rho_i(\theta)$ = new-vs-old policy ratio
* $\epsilon$ = clipping threshold

This is the same basic trust-region idea as PPO:

* if an action/output is good, raise its probability,
* if it is bad, lower its probability,
* but do not change it too aggressively in one step.

GRPO is explicitly introduced as a PPO variant, and current TRL documentation presents it as a practical trainer for this style of RL post-training.

---

## 8) Intuition for the loss

For one prompt:

* if one response gets the highest reward, it gets positive advantage,
* the model is pushed to make that response more likely next time,
* low-reward responses get negative advantage,
* the model is pushed to make them less likely.

Because the advantages are **relative within the group**, the model learns a ranking signal even if the reward scale is noisy.

That makes GRPO especially appealing when the reward is:

* binary,
* sparse,
* or only meaningful when comparing several candidate outputs. Analyses of GRPO’s behavior emphasize this group-normalized contrastive structure.

---

## 9) A tiny numeric example

Prompt:

> Solve: (2 + 3 = ?)

Sample 4 completions from the current model:

1. “The answer is 5.”
2. “It is 6.”
3. “2 + 3 = 5.”
4. “I think maybe 4.”

Suppose the reward function gives:
$$
r = [1, 0, 1, 0]
$$

### Step 1: Compute group mean

$$
\mu = \frac{1+0+1+0}{4} = 0.5
$$

### Step 2: Compute standard deviation

For this binary case:
$$
\sigma = 0.5
$$

### Step 3: Compute relative advantages

$$
A = \frac{r-\mu}{\sigma} = [1, -1, 1, -1]
$$

Interpretation:

* outputs 1 and 3 are above group average,
* outputs 2 and 4 are below group average.

### Step 4: Policy update direction

* increase probability of outputs like 1 and 3
* decrease probability of outputs like 2 and 4

That is the GRPO learning signal.

---

## 10) Why GRPO became popular for reasoning

GRPO is widely associated with reasoning-focused RL because:

* it works naturally with **verifiable rewards**,
* it avoids training a separate critic,
* it can be more memory efficient than PPO,
* and it was used in the DeepSeekMath line and later reasoning-focused training efforts.

Typical reward designs:

* exact-match answer correctness
* unit-test pass rate for code
* format reward
* reasoning-structure reward
* tool-use success
* combined weighted rewards

TRL’s GRPO trainer also supports **multiple reward functions** and combines them in practice.

---

## 11) Strengths of GRPO

### 11.1 No critic model

This reduces memory usage and implementation complexity compared with PPO-style setups.

### 11.2 Good for verifiable tasks

When you can automatically score outputs, GRPO is very natural.

### 11.3 Relative scoring can stabilize learning

Even if raw rewards are rough, group normalization can still provide a useful “better/worse than peers” signal. Recent analyses formalize this relative-group structure.

### 11.4 Strong practical momentum

GRPO has become a very visible method in reasoning-focused open post-training pipelines, and recent theory papers explicitly describe it as a widely adopted default in this area.

---

## 12) Weaknesses and pitfalls

### 12.1 All responses in a group can be bad

If every sampled answer is wrong, group-relative normalization may provide a weak or misleading signal because the “best” answer is only best **relative to other bad answers**. This issue is discussed in recent follow-up work.

### 12.2 Reward design matters a lot

GRPO is only as good as the reward function. A badly designed reward can push the model toward shortcut behavior.

### 12.3 Binary rewards can be sparse

If rewards are only 0/1 and success is rare, learning can be slow unless:

* prompts are well-curated,
* group size is chosen well,
* or extra shaping rewards are added. Recent theory papers also study how group size affects GRPO’s behavior.

### 12.4 Longer outputs may create bias

Like other RL methods on LLMs, the exact way sequence probabilities, truncation, and token aggregation are handled matters in practice.

---

## 13) GRPO vs PPO vs DPO

| Method | Data source      | Reward source                                | Critic needed | Online? | Good for                 |
| ------ | ---------------- | -------------------------------------------- | ------------- | ------- | ------------------------ |
| PPO    | model rollouts   | learned RM or rule reward                    | usually yes   | yes     | general RLHF             |
| DPO    | preference pairs | implicit from comparisons                    | no            | no      | stable offline alignment |
| GRPO   | model rollouts   | rule reward / verifier reward / reward funcs | no            | yes     | reasoning, math, code    |

### Easy intuition

* **PPO**: “I have RL rewards and a full PPO setup.”
* **DPO**: “I already have chosen vs rejected pairs.”
* **GRPO**: “I have RL rewards, but I want a lighter critic-free method using grouped rollouts.”

---

## 14) Minimal pseudocode

```python
for prompt in prompts:
    # 1) sample a group of completions from current model
    completions = [sample(model, prompt) for _ in range(G)]

    # 2) compute rewards for each completion
    rewards = [reward_fn(prompt, c) for c in completions]

    # 3) normalize rewards inside the group
    mean_r = mean(rewards)
    std_r = std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    # 4) compute logprobs under old and current policy
    # 5) form ratio rho = exp(logp_new - logp_old)
    # 6) apply PPO-style clipped objective with group-relative advantage
    # 7) update model
```

---

## 15) Minimal Hugging Face TRL-style view

TRL provides a **GRPOTrainer** specifically for this training style. The current docs describe it as support for GRPO as introduced in DeepSeekMath, and note that it can combine multiple reward functions. ([Hugging Face][8])

Conceptually, you provide:

* a model
* a prompt dataset
* one or more reward functions
* generation settings
* GRPO hyperparameters such as group size and optimization config

Then the trainer:

1. samples grouped completions,
2. computes rewards,
3. computes relative advantages,
4. updates the model.

---

## 16) The shortest mental model

GRPO says:

> For each prompt, generate several answers.
> Score them.
> Treat above-average answers as positive training signals and below-average answers as negative training signals.
> Update the model carefully using a PPO-style ratio/clipping objective.

That is the entire idea.