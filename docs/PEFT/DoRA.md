# [Weight-Decomposed Low-Rank Adaptation (DoRA)](https://arxiv.org/abs/2402.09353)

## Principle & Motivation
DoRA improves [LoRA](./LoRA.md) by **decomposing each weight into magnitude and direction, then adapting them differently.**.

Motivation: LoRA treats all directions equally → suboptimal when layers have heterogeneous importance (e.g., output layers need larger scales).

## Detailed Derivation
Base LoRA: $  W = W_0 + \Delta W  $, $  \Delta W = \frac{\alpha}{r} A B $.

DoRA decomposes pretrained W_0 into magnitude + directional unit:
$$W_0 = m_0 \odot u_0, \quad \|u_0\|_c = 1 \quad (\text{column-wise unit norm})$$
Adapter: Low-rank directional update $  \Delta u = \frac{B A}{\|B A\|_c}  $ (normalized).

Learned magnitude update:
$$m = m_0 + \Delta m \quad (\Delta m: \text{trainable vector})$$
Final:
$$W = (m_0 + \Delta m) \odot (u_0 + \Delta u)$$

Equivalent Expansion:
$$W = W_0 + \Delta m \odot u_0 + m_0 \odot \Delta u + \Delta m \odot \Delta u$$
(Last term small → often ignored, but improves expressivity).

**Why Better?** Magnitude learning mimics full-tune scale adaptation; directional low-rank captures shifts efficiently.

Empirical proof: DoRA outperforms LoRA/VeRA on commonsense/math reasoning (+2–5% average).

## Step-by-Step Code Implementation
[Python script](../../src/part3_lora_variants.ipynb)