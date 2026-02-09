# Large Models Learning
| Model ID | Key Points |
|:---:|:---:|
| [Gemma 3](./docs/Large_Models/Gemma_3.md) | ***Decoder-Only Transformer:*** [GQA]((./docs/Attention_Machanisms/GQA.md)) + 5:1 Local/Global Attention Interleaving |
| [DeepSeek-V2](./docs/Large_Models/DeepSeek_V2.md) | ***Decoder-only Transformer:*** [MLA](./docs/Attention_Machanisms/MLA.md) + [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md) |
| [DeepSeek-V3](./docs/Large_Models/DeepSeek_V3.md) |  |

# Part 1: Solidify foundations.
| Milestone                         | Status    | Notes               |
|:------------------------------------:|:-----------:|:---------------------:|
| Repo Created & Initial Commit     | Complete  | https://github.com/XP-PY/llm-prep-2026 |
| [SVD + Attention](./docs/Attention_Machanisms/SVD_Attention.md) | Complete  | Notebook + Code |
| [AdamW](./docs/Optimizer/AdamW.md) | Complete  | Notebook + Code |
| [RoPE](./docs/Position_Embeding/RoPE.md) | Complete  | Notebook + Code |

# Part 2: Master the key innovations that make modern LLMs fast and memory-efficient at scale.
| Milestone                         | Status    | Notes               |
|:------------------------------------:|:-----------:|:---------------------:|
| [FlashAttention](./docs/Attention_Machanisms/FlashAttention.md) | Complete  | Notebook |
| [GQA/MQA](./docs/Attention_Machanisms/GQA.md) | Complete  | Notebook + Code |
| [SwiGLU & RMSNorm](./docs/Activation_Layers/SwiGLU.md) | Complete  | Notebook + Code |
| [Positional Encoding Comparison](./docs/Position_Embeding/RoPE.md) | Complete  | Notebook + No compared Code |

# Part 3: Master Parameter-Efficient Fine-Tuning (PEFT) techniques that enable fine-tuning massive models on consumer GPUs
| Milestone | Status | Notes |
|:---:|:---:|:---:|
| Basic LoRA Variants: [LoRA](./docs/PEFT/LoRA.md)/[QLoRA](./docs/PEFT/QLoRA.md)/[DoRA](./docs/PEFT/DoRA.md) | Complete  | Notebook + Code |
| Specialized LoRA Variants: [LongLoRA/LoHA/VeRA](./docs/PEFT/Specialized_LoRA.md) | In Progress  | Notebook + Code |