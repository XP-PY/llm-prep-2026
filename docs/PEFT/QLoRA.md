# [Quantized LoRA (QLoRA)](https://arxiv.org/abs/2305.14314)

## Principle & Motivation
QLoRA extends [LoRA](./LoRA.md) to quantize the pretrained weights to 4-bit, enabling fine-tuning of 33B–65B models on single consumer GPUs

Multi-angle rationale:
* **Theoretical:** Pretrained weights are normally distributed → [4-bit NormalFloat (NF4)](../Math/dtypes.md) quantization preserves information better than INT4.
* **Practical:** Combines LoRA (low-rank adapters) with quantization + innovations (double quant, paged optimizers) → memory ~10x lower than FP16 LoRA.
* **Empirical:** Matches or exceeds full 16-bit fine-tune quality on instruction tasks.

## Detailed Derivation
Base LoRA: $  W = W_0 + \Delta W  $, $  \Delta W = \frac{\alpha}{r} A B $.

QLoRA additions:
1. **4-bit Quantization:**
    * Block-wise NF4: For each block of 64 weights, normalize to [-1,1], then quantize to 4-bit levels optimized for normal dist.
    * Formula: $  W_0^q = s \cdot Q_{NF4}\left( \frac{W_0 - \mu}{ \sigma } \right)  $
    (s: block scale, μ/σ: mean/var per block).
2. **Double Quantization:** Quantize the quantization constants (scales/absmax) to 8-bit → saves ~0.5 bits/param.
3. **Paged Optimizers:** Use NVIDIA unified memory + paging → optimizer states (AdamW m/v) offloaded to CPU when not needed → avoids OOM during backward.

Forward:
$$h = x \cdot \text{dequant}(W_0^q) + x \cdot \Delta W$$
(dequant on-the-fly in FP16).

**Memory Breakdown** (65B model):

* FP16 full: ~130GB.
* LoRA (r=64): ~40GB.
* QLoRA: ~10–12GB (4-bit base + low-rank FP16 adapters).

**Proof of Efficiency:** Quantization error bounded by NF4 design (theoretical info loss <1 bit/param vs uniform).

## Step-by-Step Code Implementation
[Python script](../../src/part3_lora_variants.ipynb)