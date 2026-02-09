# [Gemma 3](https://arxiv.org/abs/2503.19786)

# Overview and Introduction
* **Release Date:** March 2025 (Google DeepMind)
* **Family:** Multimodal extension of Gemma open models (building on Gemma 2 and co-designed with Gemini frontier models)
* **Sizes:** 1B, 4B, 12B, 27B parameters (lightweight, designed for consumer hardware: phones, laptops, GPUs)
* **New Capabilities:**
    * **Multimodality:** Vision understanding (images + text)
    * **Long Context:** At least 128K tokens (1B model: 32K)
    * **Multilingual:** Wider language coverage
    * **Efficiency:** Reduced KV-cache memory for long context
* **Versions:**
    * **Pre-trained (PT):** Next-token prediction
    * **Instruction-Tuned (IT):** Strong in chat, math, reasoning, multilingual, vision
* **Performance Highlights:**
    * Gemma3-4B-IT competitive with Gemma2-27B-IT
    * Gemma3-27B-IT comparable to Gemini-1.5-Pro on benchmarks

## Model Architecture (Decoder-Only Transformer)
* **Base:** Same as previous Gemma (Vaswani et al., 2017)
* **Key Changes:**
    * <span style="color: aqua">**Grouped-Query Attention (GQA):**</span> Post-norm, pre-norm with RMSNorm
    * <span style="color: aqua">**QK-Norm:**</span> Replaces soft-capping for stability (inspired by Dehghani et al., 2023; Chameleon)
    * <span style="color: aqua">**5:1 Local/Global Attention Interleaving:**</span>
        * **Pattern:** 5 local sliding-window layers (1024-token span) per 1 global full-attention layer
        * Starts with local layer
        * Reduces KV-cache memory explosion for 128K context (only global layers store full cache)
        * **RoPE:** Local = base 10k; Global = base 1M + positional interpolation (Chen et al., 2023)

# Vision Integration
* <span style="color: aqua">**Encoder:**</span> Frozen 400M SigLIP variant (Zhai et al., 2023; ViT-based, CLIP-style contrastive)
* <span style="color: aqua">**Input:**</span> Square 896×896 images
* <span style="color: aqua">**Output:**</span> Condensed to <span style="color: aqua">**fixed 256 soft tokens**</span> (embeddings) → concatenated with text tokens
* <span style="color: aqua">**Pan & Scan (P&S):**</span> Inference-only; crops non-square/high-res images into 896×896 tiles (non-overlapping, resized), encodes each → concatenates (max crops controlled)
* **Shared/frozen** across 4B/12B/27B models

## Parameter Counts

| Model | Vision Encoder | Embedding Params | Non-Embedding Params | Total Params |
|-------|---------------|-----------------|---------------------|-------------|
| 1B | 0 | 302M | 698M | 1B |
| 4B | 417M | 675M | 3.209B | 4B |
| 12B | 417M | 1.012B | 10.759B | 12B |
| 27B | 417M | 1.416B | 25.600B | 27B |

**Vocabulary:** 256k entries

# Pre-Training
* **Recipe:** Similar to Gemma 2, with distillation from larger teacher (Gemini-like)
* **Data:** Trillions of tokens (text + image-text pairs for vision)
* **Distillation:** Sample 256 candidates from teacher distribution → cross-entropy
* **Tokenizer:** Gemini 2.0 SentencePiece (262k vocab, digit splitting, whitespace preservation)

# Post-Training (Instruction Tuning - IT Models)
* **Novel Recipe:** Massive gains in math, chat, instruction-following, multilingual
* **Phases:**
    * <span style="color: aqua">**Supervised Distillation:**</span> From large IT teacher on high-quality data (integrates vision/long context)
    * <span style="color: aqua">**RL Fine-Tuning:**</span> Advanced (BOND/WARM/WARP variants) with rewards (human preferences, code execution, math ground-truth)
* **Formatting:**
    * <span style="color: aqua">**PT:**</span> Starts with BOS, ends with `<eos>`
    * <span style="color: aqua">**IT:**</span> Chat turns with `<start_of_turn>user/model`, ends `<end_of_turn>`

# Evaluations and Capabilities
* **Text-Only:** Competitive with Mixtral 8x7B, Gemini-Pro
* **Vision:** SOTA on VQA/captioning; example in Figure 1 (receipt math)
* **Long Context:** Strong needle retrieval
* **IT Strengths:** Gemma3-27B-IT ~ Gemini-1.5-Pro; 4B-IT ~ Gemma2-27B-IT
* **Quantized Versions:** int4/fp8 with minimal loss

# Limitations
* Hallucinations on rare topics
* Vision fixed-res + P&S (not native arbitrary)
* 1B limited to 32K context

# Key Takeaway
Gemma 3 advances open multimodal LLMs—efficient long-context vision in lightweight models, strong IT via novel post-training. Ideal for PyTorch experimentation on consumer hardware.