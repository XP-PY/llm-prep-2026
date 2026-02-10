# Large Models Learning
| Model ID | Key Points |
|:---:|:---:|
| [CLIP](./docs/Large_Models/CLIP.md) | ***Vision-Language Model:*** Contrastive Pre-training, zero-shot transfer, image-text encoder fusion |
| [SigLIP](./docs/Large_Models/SigLIP.md) | ***Vision-Language Model:*** Sigmoid Pairwise Loss, improved training efficiency over CLIP |
| [Gemma 3](./docs/Large_Models/Gemma_3.md) | ***Decoder-Only Transformer:*** [GQA](./docs/Attention_Machanisms/GQA.md) + 5:1 Local/Global Attention Interleaving |
| [DeepSeek-V2](./docs/Large_Models/DeepSeek_V2.md) | ***Decoder-only Transformer:*** [MLA](./docs/Attention_Machanisms/MLA.md) + [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md) |
| [DeepSeek-V3](./docs/Large_Models/DeepSeek_V3.md) | ***Decoder-only Transformer:*** [MLA](./docs/Attention_Machanisms/MLA.md) + [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md) with **auxiliary-loss-free** + Multi-token prediction (MTP) |
| [DeepSeek-VL](./docs/Large_Models/DeepSeek_VL.md) | ***Vision-Language Model (on Decoder-only LLM):*** Hybrid vision encoder (SigLIP semantic + SAM-B high-res details) → fixed-token high-res processing, gradual modality-balanced pretraining to preserve language strength |
| [DeepSeek-VL2](./docs/Large_Models/DeepSeek_VL2.md) | |

# Fine-Tuning Plan
1. **Multimodal Document Intelligence & QA System (Best Starter/High-Impact)**
    * **What:** Fine-tune DeepSeek-VL2 on a domain-specific dataset (e.g., financial reports, legal contracts, or Hong Kong government/Chinese-English bilingual documents). Add features like table/chart understanding, OCR on complex layouts, and accurate question answering.
    * **Why strong for resume:** Directly builds on the “CLIP-blind pairs” and hybrid encoder concepts we discussed. Shows you understand vision encoder limitations and how MoE models like VL2 solve them.
    * **Skills demonstrated:** LoRA fine-tuning on VL models, data curation (synthetic + real PDFs), evaluation on OCRBench/DocVQA-style metrics, Gradio demo with PDF upload.
    * **Difficulty:** Medium. Compute: VL2-Small or Tiny is very efficient.
    * **Bonus:** Deploy as a web tool and benchmark against LLaVA or closed models.

2. **VLM-Guided Text-to-Image Generation Pipeline (Directly Addresses Your Interest)**
    * **What:** Build a system where DeepSeek-VL2 first analyzes an input image or user description → generates rich, detailed prompts or scene graphs → feeds them to Janus-Pro for high-quality image generation. Add iterative refinement (e.g., “critique and regenerate” loop using R1 reasoning).
    * **Why strong:** Shows end-to-end multimodal understanding + generation in one pipeline. Addresses the exact limitation we talked about (pure CLIP-style models being “blind” to details).
    * **Skills demonstrated:** Prompt engineering with VLMs, model chaining, evaluation (GenEval, human preference studies), comparison of direct Janus-Pro vs. VL2-enhanced outputs.
    * **Difficulty:** Medium-High. Great portfolio piece with before/after image examples.
    * **Resume highlight:** “Built a multimodal pipeline that improved text-to-image alignment by X% using DeepSeek-VL2 + Janus-Pro.”

3. **Visual Reasoning Agent for Domain Tasks (Agentic + Trending)**
    * **What:** Create an agent powered by DeepSeek-R1 (reasoning) + VL2 that can take images/PDFs, reason step-by-step, and act (e.g., analyze charts and suggest data insights, or describe a UI screenshot and generate code improvements).
    * **Why strong:** Agents are hot in 2026. Combines reasoning (R1), vision (VL2), and optionally generation (Janus-Pro).
    * **Skills demonstrated:** LangGraph/LangChain or custom agent loops, tool use (image tools, code interpreter), evaluation on complex visual reasoning benchmarks.
    * **Difficulty:** Medium-High. Start simple (single image → multi-step reasoning) then add generation.

4. **Efficient Fine-Tuning & Quantization Study of DeepSeek Models**
    * **What:** Take a distilled DeepSeek-R1 or VL2-Small, apply QLoRA + 4-bit/8-bit quantization, fine-tune on a custom task (e.g., HK-specific bilingual VQA or medical image captioning), then benchmark speed/memory vs. full model.
    * **Why strong:** Demonstrates systems-level understanding (efficiency is a huge differentiator). Easy to get strong quantitative results.
    * **Skills demonstrated:** PEFT, bitsandbytes, vLLM or Ollama deployment, ablation studies.
    * **Difficulty:** Medium. Very doable on modest hardware and highly interview-discussable.

5. **Bilingual Multimodal Chat Application (Location Advantage)**
    * **What:** Fine-tune Janus-Pro or VL2 for better performance on English + Mandarin/Cantonese queries involving images (e.g., analyzing Hong Kong street signs, menus, or news screenshots).
    * **Why strong:** Shows domain adaptation and cultural awareness — especially valuable if applying to companies in Asia or with global users.
    * **Skills demonstrated:** Multilingual data synthesis, continued pre-training or SFT, evaluation on custom test set.

# LLMs Basic knowledge Learning
## Part 1: Solidify foundations.
| Milestone                         | Status    | Notes               |
|:------------------------------------:|:-----------:|:---------------------:|
| Repo Created & Initial Commit     | Complete  | https://github.com/XP-PY/llm-prep-2026 |
| [SVD + Attention](./docs/Attention_Machanisms/SVD_Attention.md) | Complete  | Notebook + Code |
| [AdamW](./docs/Optimizer/AdamW.md) | Complete  | Notebook + Code |
| [RoPE](./docs/Position_Embeding/RoPE.md) | Complete  | Notebook + Code |

## Part 2: Master the key innovations that make modern LLMs fast and memory-efficient at scale.
| Milestone                         | Status    | Notes               |
|:------------------------------------:|:-----------:|:---------------------:|
| [FlashAttention](./docs/Attention_Machanisms/FlashAttention.md) | Complete  | Notebook |
| [GQA/MQA](./docs/Attention_Machanisms/GQA.md) | Complete  | Notebook + Code |
| [SwiGLU & RMSNorm](./docs/Activation_Layers/SwiGLU.md) | Complete  | Notebook + Code |
| [Positional Encoding Comparison](./docs/Position_Embeding/RoPE.md) | Complete  | Notebook + No compared Code |

## Part 3: Master Parameter-Efficient Fine-Tuning (PEFT) techniques that enable fine-tuning massive models on consumer GPUs
| Milestone | Status | Notes |
|:---:|:---:|:---:|
| Basic LoRA Variants: [LoRA](./docs/PEFT/LoRA.md)/[QLoRA](./docs/PEFT/QLoRA.md)/[DoRA](./docs/PEFT/DoRA.md) | Complete  | Notebook + Code |
| Specialized LoRA Variants: [LongLoRA/LoHA/VeRA](./docs/PEFT/Specialized_LoRA.md) | In Progress  | Notebook + Code |
