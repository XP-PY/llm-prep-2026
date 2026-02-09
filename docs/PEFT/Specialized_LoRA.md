### Principle: Beyond Basic LoRA — Specialized Variants for Different Bottlenecks

Basic LoRA is great for general instruction tuning, but advanced variants solve specific pain points:

- **LongLoRA** — extremely long context fine-tuning with manageable memory
- **LoHA** (Low-rank Hadamard product) — higher capacity than LoRA with similar param count
- **VeRA** (Vector-based Random Matrix Adaptation) — even fewer trainable parameters (~1/10 of LoRA)
- Others (PiSSA, LoKr, etc.) — we’ll touch briefly

These appear frequently in 2024–2025 papers and production fine-tuning pipelines.

### 1. LongLoRA — Efficient Long-Context Fine-Tuning

**Core Idea**  
Standard LoRA struggles with very long sequences because attention KV cache still grows linearly with context length. LongLoRA (Chen et al., 2023) introduces:

- **Shift Short Attention** (S²-Attn): Dilated local attention during fine-tuning (only short-range attention updated via LoRA, long-range kept frozen).
- **Group-wise LoRA** (LoRA on grouped heads, similar to GQA but for adapters).

Main innovation: Only fine-tune short-range attention + LoRA on all layers → can extend LLaMA-7B from 4k to 100k+ context with ~10–20× less compute than full long-context training.

**Mathematical Core**  
During fine-tuning:
- Use shifted block-wise attention (stride > 1) for most heads.
- Apply LoRA only to these short-range computations.
- At inference: revert to full dense attention (frozen pretrained + adapted short-range).

**Formula Sketch**  
For a sequence of length L, split into blocks of size s << L.  
Attention computed only within shifted windows → O(L s) instead of O(L²).

**Real-World Impact**  
- LongLoRA-7B achieves near state-of-the-art on 100k+ Needle-in-Haystack with only ~1–2 days of fine-tuning on 8×A100.
- Used as base for many open long-context models (e.g., OpenLLaMA variants).

**Pitfalls**  
- Quality drop if shift stride too large.
- Still needs FlashAttention + GQA underneath for true efficiency.

### 2. LoHA — Higher Capacity Low-Rank via Hadamard Product

**Core Idea**  
LoRA is additive: ΔW = B A.  
LoHA (Edelman et al., 2024) uses multiplicative low-rank update:

\[
W = W_0 \odot (I + B A)
\]

or more generally Hadamard product of two low-rank matrices.

**Advantages**  
- Captures multiplicative interactions (better for some layers, e.g., value projections).
- Often achieves higher quality than LoRA at same parameter budget (r=16 LoHA ≈ r=64 LoRA in some tasks).

**Derivation**  
Let ΔU = B A (low-rank matrix).  
Then effective weight: W = W_0 ⊙ (1 + ΔU)  
→ element-wise gating of pretrained weights.

**Trade-offs**  
- Slightly higher compute (Hadamard op).
- Initialization sensitive (need careful scaling of ΔU).

### 3. VeRA — Minimal Trainable Parameters

**Core Idea** (Kopiczko et al., 2024)  
Instead of learning full low-rank matrices A and B per layer, share **global random frozen matrices** A_shared, B_shared across all layers, and learn only **very small scaling vectors** per layer.

\[
\Delta W_l = d_l^\text{down} \cdot (A_\text{shared} \cdot (B_\text{shared}^\top \cdot d_l^\text{up}))
\]

→ trainable params per layer ≈ 2 × d_model (two vectors), total ~0.01% of LoRA params.

**Surprising Result**  
VeRA often matches or exceeds LoRA quality with 10–50× fewer trainable parameters.

**When to use**  
- Extremely parameter-efficient scenarios (e.g., edge devices, massive multi-task fine-tuning).

### Comparison Table (Advanced PEFT)

| Method     | Trainable Params (relative) | Best Use Case                     | Quality vs LoRA | Memory Impact | Complexity |
|------------|-----------------------------|-----------------------------------|-----------------|---------------|------------|
| LoRA       | 100%                        | General instruction/SFT           | Baseline        | Medium        | Low        |
| LongLoRA   | ~120–150%                   | Long-context extension            | Good for long   | Lower for long seq | Medium     |
| LoHA       | ~80–120%                    | Higher capacity needed            | Often +1–4%     | Similar       | Medium     |
| VeRA       | ~1–10%                      | Ultra-low param budget            | Competitive     | Much lower    | Low        |

### Assignment (Thu–Fri)

1. **LongLoRA**  
   - Read LongLoRA paper sections 3–4.  
   - Implement shifted attention + LoRA in a small model (or use existing open impl).  
   - Fine-tune on synthetic long-copy task (train 4k, test 16k).

2. **LoHA & VeRA**  
   - Add LoHA (Hadamard) and VeRA (shared matrices + vectors) to your PEFT notebook.  
   - Compare trainable params, final perplexity, and training speed on the same dataset.

3. **Documentation**  
   - Update `Docs/peft_variants.md` with the derivations above + your ablation results.  
   - Include tables and loss curves.

4. **Commit & Reply**  
   - Push everything.  
   - Reply with your key observations (e.g., “VeRA used only 5% params of LoRA but got 92% of the performance”).

## Step-by-Step Code Implementation
[Python script](../../src/part3_lora_variants.ipynb)
