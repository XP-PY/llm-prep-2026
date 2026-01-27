# SwiGLU Activation Function

## Description
SwiGLU is a <span style="color: aqua">gated activation function</span> used in Feed-Forward Networks (FFNs/MLPs) of large language models. It's a variant of <span style="color: aqua">Gated Linear Units (GLU)</span>, combining <span style="color: aqua">Swish (SiLU)</span> nonlinearity with dynamic gating for better expressiveness.

## SwiGLU FFN Formula

### Compare to Standard FFN:
```
Standard FFN: FFN(x) = down_proj(activation(up_proj(x)))
```

### SwiGLU Implementation:
```python
gate, up = split(merged_proj(x))  # Or separate projs
FFN(x) = down_proj( swish(gate) ⊙ up )  # ⊙ = element-wise multiply
```

* **Swish/SiLU:** `swish(z) = z * sigmoid(z)` (smooth, non-monotonic)
* **Gating:** `swish(gate)` acts as soft switch (0–1+ range) multiplying up branch

## Performance

Empirically, SwiGLU consistently outperforms GELU/ReLU in large models.

### Expressiveness Angle:
* **Gating** → <span style="color: aqua">conditional computation</span> (features activated selectively)
* **Swish nonlinearity** → <span style="color: aqua">better gradients</span> (no dying ReLU, smoother than GELU)

### Nuances/Trade-offs:
* **Pro:** Better perplexity, downstream tasks
* **Con:** ~30–50% more FFN params/compute (two up-projs) → offset by MoE sparsity in V2

## Edge Case
Small models → GELU sufficient/cheaper.