# [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)

## Principle: Why LoRA Revolutionized Fine-Tuning
Full fine-tuning: Update all billions of parameters → massive memory (e.g., 70B FP16 = 140GB).
> * Total number of model parameters × Number of bytes per parameter = Total memory requirement
>   * Parameters number: 70B = 70 × 10^9
>   * FP16 byte: FP16 = 2 bytes (16 bits ÷ 8 bits/byte = 2 bytes)
>   * Memory requirement of Weight: 70 × 10^9 Parameters × 2 byte/parameter = 140 × 10^9 bytes ≈ 130.4 GB (140 × 10^9 bytes ÷ 1024^3)
>
> * Some other part in memory when Training/Inference:
>   * gradients_memory (Training period)
>   * optimizer_states_memory
>   * activations_memory (Forward propagation)
>   * kv_cache_memory (Inference)
>   * ...

**Key Empirical Insight:** In pretrained models, task-specific updates $\Delta W$ have low intrinsic rank — [SVD shows rapid singular value decay](../Attention_Machanisms/SVD_Attention.md)

Theoretical angles:

* Pretrained weights capture general features → adaptation needs only "directions" (low-rank subspace).
* Overparameterization: Transformers have redundant dimensions → updates concentrate in few modes.

Proof sketch: Assume loss landscape near pretrained minimum is flat in most directions → gradients/updates span low-dimensional manifold.

## Detailed Mathematical Formulation
For a pretrained weight matrix $  W_0 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}  $ (e.g., attention projection).

LoRA parameterizes update as:$$W = W_0 + \Delta W, \quad \Delta W = \frac{\alpha}{r} B A^T$$where:
* $  A \in \mathbb{R}^{d_{\text{in}} \times r}  $ (down-projection, random Gaussian init).
* $  B \in \mathbb{R}^{d_{\text{out}} \times r}  $ (up-projection, zero init → $\Delta W=0$ at start).
* $  r \ll \min(d_{\text{in}}, d_{\text{out}})  $ (rank, e.g., 8–64).
* $\alpha$: Scaling Factor. To stabilize training (avoid small gradients from low $r$)

Forward pass:
$$h = x W = x W_0 + x \Delta W = x W_0 + \frac{\alpha}{r} (x A) B^T$$

## Comparison Table
|Variant|Update Form|Trainable Params (per matrix)|Key Innovation|Quality vs Full|Memory (70B Fine-Tune)|Used In|When should use what|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Full|W_new|d_in d_out|None|Best|1TB+|Rare at scale|-|
|LoRA|$W_0+BA$|r(d_in + d_out)|Low-rank factorization|Near full|~40GB|Alpaca, most PEFT|default choice; simplest; strong baseline; good when you can load the base in fp16/bf16|
|[QLoRA](./QLoRA.md)|dequant(W_0^{4bit}) + B A|Same + quant overhead|4-bit + paging|Competitive|~10GB|Guanaco, academic|choose when VRAM is tight and you want to fine-tune bigger bases by quantizing them to 4-bit while training adapters|
|[DoRA](./DoRA.md)|(W_0 + unit-norm(B A)) ⊙ m|Same + magnitude vector|Magnitude-direction split|Often > LoRA|Similar|Latest (2024+) SOTA|try when LoRA underperforms full FT more than you’d like; it aims to narrow that gap while staying PEFT and keeping inference overhead minimal|

## Step-by-Step Code Implementation
[Python script](../../src/part3_lora_variants.ipynb)