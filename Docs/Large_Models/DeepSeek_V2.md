# DeepSeek-V2
![Model_Architecture](../Resource/pics/DeepSeek-V2.png)

# Paper Overview & Key Highlights

* **Title/Author:** DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (DeepSeek-AI, June 2024).
* **Model Scale:** 236B total parameters, <span style="color: aqua">21B activated per token</span> (sparse MoE), 128K context length.
* **Core Innovations (Two main for efficiency):**
    * <span style="color: aqua">**Multi-head Latent Attention (MLA)**</span> — Boosting inference efficiency.
    * <span style="color: aqua">**DeepSeekMoE**</span> — Training strong models at economical costs.
* **Wins vs. DeepSeek 67B (dense predecessor):**
    * Stronger performance (top-tier open-source).
    * 42.5% lower training cost.
    * 93.3% smaller KV cache.
    * 5.76× higher max generation throughput.

# Overall Architecture

* **Decoder-only Transformer** with two modifications:
    * **Attention layers:** MLA (instead of MHA).
    * **FFN layers:** DeepSeekMoE (instead of dense FFN).
* **[MLA](../Attention_Machanisms/MLA.md):**
    * **Role:** Efficient inference via KV compression + decoupled RoPE.
* **[DeepSeekMoE](../MoE/DeepSeekMoE.md):**
    * **Role:** Economical training via sparse routed experts + shared backbone.

# Pre-Training (Section 3)

* **Data:** 8.1T high-quality tokens (multi-source, heavy Chinese/English balance, aggressive dedup/filter/debias).
    * Jump from DeepSeek-LLM's 2T → more capacity fill for larger MoE.
* **Setups:**
    * Hyperparams tuned for MoE stability.
    * Expert parallelism (EP) for routed experts.
    * Long context extension to 128K (MLA enables).
* **Evaluations:**
    * **Base model:** Top open-source (MMLU leader with few active params).
    * **Efficiency:** Training cheaper, inference faster (paper Figure 1b).

# Alignment (Section 4)

* **SFT:** 1.5M multi-turn sessions (diverse domains: math/code/reasoning/safety/multilingual).
* **RL:** GRPO (Group Relative Policy Optimization) for preference alignment.
* **Chat Versions:** DeepSeek-V2 Chat (SFT) and Chat (RL) — helpful/harmless, strong multilingual.

# Evaluations & Results (Sections 3–4)

* **Benchmarks:** English/Chinese (MMLU/CMMLU, math/code like GSM8K/MATH/HumanEval, long-context).
* **Highlights:**
    * **Base:** Beats many dense with more active params.
    * **Chat:** Competitive/top open-source (MT-Bench, Arena-Hard).
    * **Strengths:** Math/code/reasoning (DeepSeek tradition), bilingual.
* **Efficiency Table (vs. DeepSeek 67B dense):**

| Metric | DeepSeek 67B | DeepSeek-V2 |
| :--- | :--- | :--- |
| Training Cost | Baseline | 57.5% (42.5% save) |
| KV Cache | Baseline | 6.7% (93.3% reduction) |
| Max Throughput | Baseline | 5.76× higher |

# Conclusion, Limitations, Future Work (Section 5)

* **Conclusion:** Proves MoE + smart attention viable for open-source scaling (strong/economical/efficient).
* **Limitations:**
    * Hallucinations (niche facts).
    * Safety (edge jailbreaks).
    * VRAM for all experts (quant helps).
    * MoE routing noise on OOD.
* **Future:** Larger scale, better long-context/safety, variants (e.g., V2-Lite 16B Appendix B).

# Appendices (Quick References)

* **B:** DeepSeek-V2-Lite (16B MoE with same innovations).
* **D:** MLA ablations (beats MHA/GQA/MQA).
* **E:** Data debiasing details.
* **Others:** Math/code extras, formats.