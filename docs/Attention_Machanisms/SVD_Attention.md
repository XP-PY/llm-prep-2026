# Singular Value Decomposition (SVD) on Attention Matrices

## Principle: Why Interpret Attention via SVD?
Attention in Transformers (from "Attention is All You Need") computes weighted averages of values based on query-key similarities. The attention weight matrix $$A = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d}} \right) $$ (shape: $  n \times n  $, n=sequence length) is row-stochastic (rows sum to 1), capturing token relationships.

> A square matrix $  A \in \mathbb{R}^{n \times n}  $ is row-stochastic if:
> 1. Every entry is non-negative: $  A_{ij} \geq 0 \ \forall i,j  $
> 2. Each row sums to exactly 1: $  \sum_{j=1}^n A_{ij} = 1 \ \forall i  $
>
> This means each row can be interpreted as a **probability distribution** over the columns.

[SVD (Singular Value Decomposition)](../Math/SVD.md) interprets A by factoring it into $  A = U \Sigma V^T  $:
* $  U, V  $: Orthogonal matrices (left/right singular vectors—directions of input/output spaces).
* $  \Sigma  $: Diagonal matrix of singular values $  \sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_n  $ (magnitudes).

**Key Insight**: The singular value spectrum (plot of $  \sigma_i  $) reveals A's <span style="color: aqua">effective rank</span>—how many "independent directions" it uses. In trained LLMs, spectra decay rapidly (e.g., first 10-20 $  \sigma_i  $ capture 99% variance), meaning low rank ($  \operatorname{rank}(A) \ll n  $). This implies attention is redundant/expressible with fewer computations, justifying compressions like LoRA.

### Theoretical Derivation (Multi-Angle):
From linear algebra: Any matrix A has SVD, but for attention:
1. **Expressivity**: High-rank A allows diverse mixtures (rich token interactions). Low-rank means "bottlenecks" but efficiency.
2. **Why Low-Rank in Practice?** Empirical (e.g., "Low-Rank Adaptation" paper): Training induces sparsity; later layers focus on few heads. Derivation hint: Attention approximates a kernel $  K(x_i, x_j)  $, and low-rank kernels are decomposable.$$A \approx \sum_{k=1}^r \sigma_k u_k v_k^T \quad (r \ll n, \text{low-rank approx})$$
3. **Pitfalls**: SVD on softmax-ed A is numerical (not closed-form), sensitive to masking (causal attention).
4. **Real-World Cases**: In LLaMA-3, low-rank attention enables QLoRA (fine-tune 70B on 24GB GPU). DeepSeek-MoE compresses KV cache via SVD-like methods, saving 50% memory.
5. **Future Implications**: As models scale (e.g., Grok-2 MoE), SVD informs "rank-aware" training—prune low $  \sigma_i  $ early for faster convergence. Check arXiv: "Spectral Analysis of Self-Attention" (2023) for latest.

## Step-by-Step Code Implementation
[Python script](../../src/week1_svd_attention.ipynb)