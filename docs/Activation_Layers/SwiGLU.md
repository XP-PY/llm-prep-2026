# SwiGLU Activation & RMSNorm

## Principle: Why These Replacements Dominate Modern LLMs
Standard Transformer FFN: GeLU(SiLU) activation + two linear layers.

Issues at scale:
* [GeLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html): Computationally heavy| less expressive in deep nets.
* [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html): Mean/variance computation → slow on GPUs| sensitive to batch stats.

[**SwiGLU**](https://arxiv.org/abs/2002.05202): Gated linear unit with Swish (SiLU) — better gradient flow| higher capacity.

[**RMSNorm**](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html): Normalize by root mean square only — no centering| ~10-20% faster| equivalent stability.

**Comparison Table**
|Component|Standard (Original Transformer)|Modern Replacement|Params Increase|Speed Impact|Quality Impact|Used In|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Activation|GeLU/ReLU|**SwiGLU**|~50% (extra gate)|Slight slower compute|Significant ↑|LLaMA, PaLM, Gemma|
|Normalization|LayerNorm (mean + var)|**RMSNorm**|None|10-20% faster|Equivalent/↑| LLaMA, GPT-NeoX, Falcon |

## Detailed Derivations

### SwiGLU Derivation

#### Description
SwiGLU is a <span style="color: aqua">gated activation function</span> used in Feed-Forward Networks (FFNs/MLPs) of large language models. It's a variant of <span style="color: aqua">Gated Linear Units (GLU)</span>| combining <span style="color: aqua">Swish (SiLU)</span> nonlinearity with dynamic gating for better expressiveness.

#### SwiGLU FFN Formula

##### Compare to Standard FFN:
```
Standard FFN: FFN(x) = down_proj(activation(up_proj(x)))
```

##### SwiGLU Implementation:
```python
gate, up = split(merged_proj(x))  # Or separate projs
FFN(x) = down_proj( swish(gate) ⊙ up )  # ⊙ = element-wise multiply
```

* **[Swish/SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html):** `swish(z) = z * sigmoid(z)` (smooth, non-monotonic → richer representations)
* **Gating:** `swish(gate)` acts as soft switch (0–1+ range) multiplying up branch

#### Performance

Empirically, SwiGLU consistently outperforms GELU/ReLU in large models.

##### Expressiveness Angle:
* **Gating** → <span style="color: aqua">conditional computation</span> (features activated selectively)
* **Swish nonlinearity** → <span style="color: aqua">better gradients</span> (no dying ReLU, smoother than GELU)

##### Nuances/Trade-offs:
* **Pro:** Better perplexity, downstream tasks
* **Con:** ~30–50% more FFN params/compute (two up-projs) → offset by MoE sparsity in V2

#### Edge Case
Small models → GELU sufficient/cheaper.

### RMSNorm Derivation
**LayerNorm:**
$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
where μ = mean(x), σ² = var(x).

**RMSNorm (re-parameterized):**
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum x_i^2 + \epsilon}} \odot \gamma$$
(No β bias, as centering unnecessary post-initialization.)

**Theoretical:** Removes mean subtraction → equivalent in trained models (weights absorb shift). Faster: No mean pass.

Proof sketch: In deep nets, activations center around 0 naturally; variance normalization suffices.

## Step-by-Step Code Implementation
[Python script](../../src/part2_swiglu_rmsnorm.ipynb)