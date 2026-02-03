# Grouped-Query Attention (GQA)

## Motivation: Inference Memory Bottleneck in Autoregressive Decoding
During inference (autoregressive generation):

* **Q**: Computed for current token only (1 × heads × head_dim).
* **K/V**: Cached from all previous tokens → KV cache size dominates memory.
* **Standard MHA**: Separate K/V per head → cache size = 2 × layers × seq_len × heads × head_dim × bytes_per_param.

**Goal**: Reduce unique K/V projections while preserving most of MHA's expressivity.

* **MQA (extreme)**: 1 shared K/V → cache ÷ heads.
* **GQA (balanced)**: G shared K/V groups → cache ÷ (heads / G).

**Theoretical angle**: Attention heads specialize (some for syntax, some semantics). Sharing K/V forces "group consensus" → minor quality drop but huge efficiency win.

## Detailed Mathematical Formulation
Let:

* $  d_{\text{model}}  $: Embedding dimension (e.g., 4096).
* $  h  $: Number of query heads (e.g., 32).
* $  g  $: Number of KV groups (e.g., 8 for LLaMA-3 80B).
* $  d_k = d_v = d_{\text{model}} / h  $: Head dimension (e.g., 128).

**Standard MHA** Projections:

* Query: $  Q = X W_Q  $, $  W_Q \in \mathbb{R}^{d_{\text{model}} \times h d_k}  $.
* Key: $  K = X W_K  $, $  W_K \in \mathbb{R}^{d_{\text{model}} \times h d_k}  $ (separate per head).
* Value: $  V = X W_V  $, $  W_V \in \mathbb{R}^{d_{\text{model}} \times h d_v}  $.

Reshape: $  Q \to (B, T, h, d_k)  $, same for K/V.

**MQA (g=1)**:

* $  W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}  $ (shared, no head dim).
* K/V reshaped to (B, T, 1, d_k) → broadcast/repeat to h heads.

**GQA (general g)**:

* Number of unique KV heads = g.
* $  W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times g d_k}  $.
* K reshaped to (B, T, g, d_k).
* To match Q heads: Repeat each KV group across (h / g) query heads:$$K_{\text{repeated}} = K[:, :, \operatorname{repeat}(g \to h), :]$$(broadcast: each of g KV heads duplicated h/g times).

Attention scores (unchanged):
$$\text{Attn}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K_{\text{repeated}}^T}{\sqrt{d_k}} \right) V_{\text{repeated}}$$

**Parameter Count Comparison** (per layer, ignoring biases):
$$\begin{array}{|c|c|c|}
\hline
\text{Variant} & \text{Total Params (QKV)} & \text{KV Cache Multiplier} \\
\hline
\text{MHA} & 3 d_{\text{model}}^2 & 1 \times \\
\text{MQA (g=1)} & d_{\text{model}}^2 + 2 d_{\text{model}} (d_{\text{model}}/h) & 1/h \times \\
\text{GQA (g)} & d_{\text{model}}^2 + 2 d_{\text{model}} (d_{\text{model}} g / h) & g/h \times \\
\hline
\end{array}$$
For LLaMA-3 8B (d=4096, h=32, g=8): KV cache ~4x smaller than MHA.

**Derivation of Broadcasting**:
* Let group size s = h / g.
* K shape after projection: (B, T, g, d_k).
* Insert dim: (B, T, 1, g, d_k) → repeat s times on dim=2 → (B, T, s, g, d_k) → reshape to (B, T, h, d_k).

This ensures each query head group attends to its shared KV.

## Real-World Cases and Ablations

* **LLaMA-3 (Meta, 2024)**: 8B uses g=8 (4x KV reduction vs MHA), near-MHA perplexity. 70B uses g=8 early layers, g=1 later (hybrid for speed).
* **Gemma-2**: Similar GQA for balanced quality/speed.
* **Pitfalls**: Too low g (e.g., MQA) → quality drop on reasoning tasks (heads lose specialization). Too high g → no speedup.
* **Empirical**: Ainslie et al. (2023) showed GQA matches MHA on most benchmarks with 2-3x faster decoding.
* **Future Implications**: Combine with Sliding Window Attention (SWA) or MLA (Multi-Linear Attention) for even longer contexts; key for multimodal (e.g., video LLMs).

## Step-by-Step Code Implementation
[Python script](../../src/part2_gqa.ipynb)