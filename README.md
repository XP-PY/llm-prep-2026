# Large Models Learning
| Model ID | Key Points |
|:---:|:---:|
| [CLIP](./docs/Large_Models/CLIP.md) | ***Vision-Language Model:*** Contrastive Pre-training, zero-shot transfer, image-text encoder fusion |
| [SigLIP](./docs/Large_Models/SigLIP.md) | ***Vision-Language Model:*** Sigmoid Pairwise Loss, improved training efficiency over CLIP |
| [Gemma 3](./docs/Large_Models/Gemma_3.md) | ***Vision-Language Model (on Decoder-only LLM):*** vision encoder (SigLIP) + [GQA](./docs/Attention_Machanisms/GQA.md) + 5:1 Local/Global Attention Interleaving |
| [DeepSeek-V2](./docs/Large_Models/DeepSeek_V2.md) | ***Decoder-only Transformer:*** [MLA](./docs/Attention_Machanisms/MLA.md) + [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md) |
| [DeepSeek-V3](./docs/Large_Models/DeepSeek_V3.md) | ***Decoder-only Transformer:*** [MLA](./docs/Attention_Machanisms/MLA.md) + [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md) with **auxiliary-loss-free** + Multi-token prediction (MTP) |
| [DeepSeek-V3.2](./docs/Large_Models/DeepSeek_V32.md) | |
| [DeepSeek-VL](./docs/Large_Models/DeepSeek_VL.md) | ***Vision-Language Model (on Decoder-only LLM):*** Hybrid vision encoder (SigLIP semantic + SAM-B high-res details) → fixed-token high-res processing, gradual modality-balanced pretraining to preserve language strength |
| [DeepSeek-VL2](./docs/Large_Models/DeepSeek_VL2.md) | ***Vision-Language Model (MoE Decoder-only LLM):*** Single SigLIP dynamic tiling (global thumbnail + local tiles) → arbitrary high-res/aspect ratios with controlled tokens, DeepSeekMoE backbone with MLA |

# LLM Knowledge System
A topic-based map of this repo. This section is organized by knowledge domains rather than learning phases.

## Visual Map
```mermaid
flowchart TD
    A[LLM Knowledge System]

    A --> B[Foundations]
    A --> C[Architecture and Scaling]
    A --> D[Adaptation and Alignment]
    A --> E[Inference and Serving]
    A --> G[Model Case Studies]

    B --> B1[SVD / dtypes / AdamW]
    B --> B2[Attention: MHA / MQA / GQA]
    B --> B3[RoPE / SwiGLU]

    C --> C1[FlashAttention / MLA]
    C --> C2[DeepSeekMoE]
    C --> C3[TP / PP / EP]

    D --> D1[LoRA / QLoRA / DoRA]
    D --> D2[Specialized LoRA Variants]
    D --> D3[SFT / RLHF / DPO / PPO / GRPO]

    E --> E1[Speculative Decoding]
    E --> E2[Continuous Batching / PagedAttention]
    E --> E3[AWQ / GPTQ / TensorRT-LLM]
    E --> E4[Hallucination Mitigation]

    G --> G1[DeepSeek-V2]
    G --> G2[DeepSeek-V3]
    G --> G3[DeepSeek-V3.2]

    G1 -. combines .-> C1
    G1 -. combines .-> C2
    G2 -. combines .-> C1
    G2 -. combines .-> C2
```

The diagram gives a high-level overview; the sections below act as the detailed index.

| Domain | Focus | Core Topics |
|:---|:---|:---|
| Foundations | Math, optimization, and Transformer building blocks | SVD, dtypes, AdamW, MHA/MQA/GQA, RoPE, SwiGLU |
| Architecture & Scaling | Efficient training and large-scale model design | FlashAttention, MLA, DeepSeekMoE, TP/PP/EP |
| Adaptation & Alignment | Task adaptation and preference learning | LoRA family, SFT, RLHF, DPO, PPO, GRPO |
| Inference & Serving | Latency, memory, and deployment efficiency | Speculative Decoding, Continuous Batching, Quantization, TensorRT-LLM, Hallucination Mitigation |

## 1. Foundations
- Math and numerical basics: [SVD](./docs/Math/SVD.md), [dtypes](./docs/Math/dtypes.md), [AdamW](./docs/Optimizer/AdamW.md)
- Attention mechanisms: [SVD + Attention](./docs/Attention_Machanisms/SVD_Attention.md), [MHA](./docs/Attention_Machanisms/MHA.md), [MQA](./docs/Attention_Machanisms/MQA.md), [GQA](./docs/Attention_Machanisms/GQA.md)
- Position and FFN blocks: [RoPE](./docs/Position_Embeding/RoPE.md), [SwiGLU](./docs/Activation_Layers/SwiGLU.md)

## 2. Architecture & Scaling
- Efficient attention: [FlashAttention](./docs/Attention_Machanisms/FlashAttention.md), [MLA](./docs/Attention_Machanisms/MLA.md)
- Sparse architecture: [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md)
- Distributed training: [TP](./docs/Parallelism/TP.md), [PP](./docs/Parallelism/PP.md), [EP](./docs/Parallelism/EP.md) (`In Progress`)

## 3. Adaptation & Alignment
- PEFT: [LoRA](./docs/PEFT/LoRA.md), [QLoRA](./docs/PEFT/QLoRA.md), [DoRA](./docs/PEFT/DoRA.md), [Specialized LoRA Variants](./docs/PEFT/Specialized_LoRA.md)
- Supervised and preference alignment: [SFT](./docs/Preference_Alignment/SFT.md), [RLHF](./docs/Preference_Alignment/RLHF.md), [DPO](./docs/Preference_Alignment/DPO.md), [PPO](./docs/Preference_Alignment/PPO.md), [GRPO](./docs/Preference_Alignment/GRPO.md)

## 4. Inference & Serving
- Decoding acceleration: [Speculative Decoding (Medusa/Lookahead)](./docs/Inference_Optimization/speculative_decoding.md)
- Serving systems: [Continuous Batching & PagedAttention](./docs/Inference_Optimization/continuous_batching.md), [TensorRT-LLM & Multi-LoRA Serving](./docs/Inference_Optimization/tensorrt_multilora.md)
- Compression and reliability: [Post-Training Quantization (AWQ/GPTQ)](./docs/Inference_Optimization/quantization_inference.md), [Hallucination Mitigation at Inference](./docs/Inference_Optimization/hallucination_mitigation.md)

# File Structure of /docs
```text
docs/
|-- Activation_Layers/
|   `-- SwiGLU.md
|-- Attention_Machanisms/
|   |-- FlashAttention.md
|   |-- GQA.md
|   |-- MHA.md
|   |-- MLA.md
|   |-- MQA.md
|   `-- SVD_Attention.md
|-- Inference_Optimization/
|   |-- continuous_batching.md
|   |-- hallucination_mitigation.md
|   |-- quantization_inference.md
|   |-- speculative_decoding.md
|   `-- tensorrt_multilora.md
|-- Large_Models/
|   |-- CLIP.md
|   |-- DeepSeek_V2.md
|   |-- DeepSeek_V3.md
|   |-- DeepSeek_V32.md
|   |-- DeepSeek_VL.md
|   |-- DeepSeek_VL2.md
|   |-- Gemma_3.md
|   `-- SigLIP.md
|-- Math/
|   |-- SVD.md
|   `-- dtypes.md
|-- MoE/
|   `-- DeepSeekMoE.md
|-- Optimizer/
|   `-- AdamW.md
|-- PEFT/
|   |-- DoRA.md
|   |-- LoRA.md
|   |-- QLoRA.md
|   `-- Specialized_LoRA.md
|-- Parallelism/
|   |-- EP.md
|   |-- PP.md
|   `-- TP.md
|-- Position_Embeding/
|   `-- RoPE.md
|-- Preference_Alignment/
|   |-- DPO.md
|   |-- GRPO.md
|   |-- PPO.md
|   |-- RLHF.md
|   `-- SFT.md
`-- Resource/
    |-- Text_Color_Table.md
    `-- pics/
        |-- DeepSeek-V2.png
        |-- DeepSeek-V3_ALF.png
        |-- DeepSeek-V3_MTP.png
        |-- DeepSeek-V3_MoE.png
        |-- DeepSeek-VL.png
        |-- DeepSeek-VL2_fig1.png
        |-- DeepSeek-VL2_fig2.png
        |-- DeepSeek-VL2_fig3.png
        |-- DeepSeek-VL32_Attention_Architecture.png
        |-- DeepSeek-VL32_Inference-cost.png
        `-- DeepSeek-VL32_MHA-and-MQA-modes-of-MLA.png
```

# Learning Resource Recommendation
- [LLM 八股文](https://my.feishu.cn/wiki/XGkRwrugwisqaokx909caQ4anEb)

[//]: # "Archived previous phase-based version of the README section:"
[//]: # ""
[//]: # "# LLMs Basic knowledge Learning"
[//]: # "- ## Part 1: Solidify foundations."
[//]: # "    | Milestone                         | Status    | Notes               |"
[//]: # "    |:------------------------------------:|:-----------:|:---------------------:|"
[//]: # "    | Repo Created & Initial Commit     | Complete  | https://github.com/XP-PY/llm-prep-2026 |"
[//]: # "    | [SVD + Attention](./docs/Attention_Machanisms/SVD_Attention.md) | Complete  | Notebook + Code |"
[//]: # "    | [AdamW](./docs/Optimizer/AdamW.md) | Complete  | Notebook + Code |"
[//]: # "    | [RoPE](./docs/Position_Embeding/RoPE.md) | Complete  | Notebook + Code |"
[//]: # ""
[//]: # "- ## Part 2: Master the key innovations that make modern LLMs fast and memory-efficient at scale."
[//]: # "    | Milestone                         | Status    | Notes               |"
[//]: # "    |:------------------------------------:|:-----------:|:---------------------:|"
[//]: # "    | [FlashAttention](./docs/Attention_Machanisms/FlashAttention.md) | Complete  | Notebook |"
[//]: # "    | [GQA/MQA](./docs/Attention_Machanisms/GQA.md) | Complete  | Notebook + Code |"
[//]: # "    | [SwiGLU & RMSNorm](./docs/Activation_Layers/SwiGLU.md) | Complete  | Notebook + Code |"
[//]: # "    | [Positional Encoding Comparison](./docs/Position_Embeding/RoPE.md) | Complete  | Notebook |"
[//]: # ""
[//]: # "- ## Part 3: Master Parameter-Efficient Fine-Tuning (PEFT) techniques that enable fine-tuning massive models on consumer GPUs"
[//]: # "    | Milestone | Status | Notes |"
[//]: # "    |:---:|:---:|:---:|"
[//]: # "    | Basic LoRA Variants: [LoRA](./docs/PEFT/LoRA.md)/[QLoRA](./docs/PEFT/QLoRA.md)/[DoRA](./docs/PEFT/DoRA.md) | Complete  | Notebook + Code |"
[//]: # "    | Specialized LoRA Variants: [LongLoRA/LoHA/VeRA](./docs/PEFT/Specialized_LoRA.md) | Complete  | Notebook + Code |"
[//]: # "    | Preference Alignment: [SFT](./docs/Preference_Alignment/SFT.md)/[RLHF](./docs/Preference_Alignment/RLHF.md)/[DPO](./docs/Preference_Alignment/DPO.md) | Complete  | Notebook |"
[//]: # ""
[//]: # "- ## Part 4: Inference Optimization Mastery"
[//]: # "    | Milestone | Status | Notes |"
[//]: # "    |:---:|:---:|:---:|"
[//]: # "    | [Speculative Decoding (Medusa/Lookahead)](./docs/Inference_Optimization/speculative_decoding.md) for Inference Speedups | Complete | Notebook |"
[//]: # "    | [Continuous Batching & PagedAttention](./docs/Inference_Optimization/continuous_batching.md) | Complete | Notebook |"
[//]: # "    | [Post-Training Quantization (AWQ/GPTQ)](./docs/Inference_Optimization/quantization_inference.md) | Complete | Notebook |"
[//]: # "    | [TensorRT-LLM & Multi-LoRA Serving](./docs/Inference_Optimization/quantization_inference.md) | Complete | Notebook |"
[//]: # "    | [Hallucination Mitigation at Inference](./docs/Inference_Optimization/hallucination_mitigation.md) | Complete | Notebook |"
[//]: # ""
[//]: # "- ## Part 5: Multimodality"
[//]: # ""
[//]: # "- ## Part 6: Others"
[//]: # "    | Milestone | Status | Notes |"
[//]: # "    |:---:|:---:|:---:|"
[//]: # "    | Parallel Methods: [TP](./docs/Parallelism/TP.md)/[PP](./docs/Parallelism/PP.md)/[EP](./docs/Parallelism/EP.md) | In Progress | Notebook |"
