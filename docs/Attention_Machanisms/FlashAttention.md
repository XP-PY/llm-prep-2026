# FlashAttention

## Core Motivation: GPU Memory Hierarchy and IO Bottlenecks
GPUs have two memory levels:

* **HBM** (High-Bandwidth Memory): Large (40–80 GB on A100/H100), slow accesses (~2 TB/s but high latency).
* **SRAM** (Shared Memory per SM): Small (~100 KB per block), fast (10–20 TB/s, low latency).

**Standard attention:**

* Compute full $  S = \frac{Q K^T}{\sqrt{d}}  $ → write to HBM: O(n²) memory.
* Softmax(S) → write full A to HBM.
* O = A @ V → read A back from HBM.

**FlashAttention:** Never materializes full S or A. Computes everything in SRAM via tiling (block-wise processing) and online softmax (running max/sum statistics). Memory drops to O(n), exact arithmetic.
Multi-angle view:

* **Theoretical:** Reduces IO from O(n²) to O(n) by fusing operations (kernel fusion).
* **Practical:** Enables 32k–128k contexts in training/inference (LLaMA-3, Grok).
* **Pitfalls:** Requires careful block sizing (B_r, B_c) based on head_dim and SRAM limits (~64–512 typical).
* **Future:** Flash-3 adds quantization (FP8) and head_dim partitioning for H100 → 700 TFLOPs utilization.

## Forward Pass Derivation: Tiled Online Softmax
Split into tiles:
* Q tiled into blocks of size $  B_r  $ (by rows).
* K/V tiled into blocks of size $  B_c  $ (by rows).
* For causal attention: Process Q block i with K/V blocks j ≤ i.

For each Q block i (shape: $  B_r \times d  $):

Initialize per-row statistics (Init in HBM, Compute in SRAM):
* $  m_i  $: Running maximum (for numerical stability).
* $  l_i  $: Running sum of exponentials (normalizer).
* $  O_i  $: Running output accumulator.

**Initial (before any K block):** $  m_i = -\infty  $, $  l_i = 0  $, $  O_i = 0  $.

For each compatible K/V block j:
1. Load $  K_j  $ ($  B_c \times d  $), $  V_j  $ ($  B_c \times d  $) into SRAM.
2. Compute local scores: $  S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}  $ ($  B_r \times B_c  $, in SRAM).
3. Compute local row max: $  m_{ij} = \operatorname{rowmax}(S_{ij})  $ (vector $  B_r \times 1  $).
4. Update global max: $  m_i^{\text{new}} = \max(m_i^{\text{old}}, m_{ij})  $, where $m_i^{\text{old}}$ is the maximum value of that row in the previously processed block.
5. Rescale previous contributions (avoid underflow):
    * Old scaling factor: $  e^{m_i^{\text{old}} - m_i^{\text{new}}}  $.
6. Compute new exponentials: $  P_{ij} = \exp(S_{ij} - m_i^{\text{new}})  $ (unnormalized probs, in SRAM).
7. Update normalizer:$$l_i^{\text{new}} = e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot l_i^{\text{old}} + \operatorname{rowsum}(P_{ij})$$
8. Update output accumulator:$$O_i^{\text{new}} = e^{m_i^{\text{old}} - m_i^{\text{new}}} \cdot O_i^{\text{old}} + P_{ij} @ V_j$$
9. Store updated $  m_i^{\text{new}}, l_i^{\text{new}}, O_i^{\text{new}}  $ (small vectors, O(B_r), write back to HBM).

After all j blocks for this i:
$$O_i = \frac{O_i^{\text{new}}}{l_i^{\text{new}}}$$
**Why this works:** Equivalent to full softmax because rescaling preserves ratios (mathematical proof in paper: telescoping exponentials cancel).

LaTeX full per-tile update:
$$m_i^{(j)} = \max\left(m_i^{(j-1)}, \max_k S_{ij,k}\right)$$
$$l_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} l_i^{(j-1)} + \sum_k e^{S_{ij,k} - m_i^{(j)}}$$
$$O_i^{(j)} = e^{m_i^{(j-1)} - m_i^{(j)}} O_i^{(j-1)} + \left( e^{S_{ij} - m_i^{(j)}} \right) V_j$$

## Backward Pass: Recomputation for Memory Savings
The backward pass needs to compute the gradients for `Q`, `K`, and `V`.

*   **Standard Approach:** Requires the intermediate matrices `S` (attention scores) and `P` (attention weights after softmax).
*   **FlashAttention Approach:**

    1.  The forward pass only stores the final output `O` and the summary statistics `l` (log-sum-exp) and `m` (row-wise maximum).
    2.  During backpropagation, for each block `Q_i`, we **re-load the corresponding blocks of `K` and `V`** and **recompute the forward pass's tiled computation**. This dynamically reconstructs all intermediate values needed to compute `S_ij` and `P_ij` within SRAM. (The stored `l` and `m` allow for the exact recovery of the softmax normalization factors.)
    3.  Using the dynamically reconstructed `P_ij` and the upstream gradient `dO_i`, we compute the gradients `dQ_i`, `dK_j`, and `dV_j` in SRAM.
    4.  These gradients are then accumulated and written back to HBM.

## Step-by-Step Code Implementation
[Python script](../../src/part2_flash_attention.ipynb)