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
    C --> C3[TP / PP / EP / FSDP]
    C --> C4[Gradient Checkpointing / Mixed Precision]

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
| Foundations | Math, optimization, losses, normalization, and Transformer building blocks | SVD, dtypes, AdamW, learning rate schedulers, Sigmoid, GELU, LayerNorm, RMSNorm, BatchNorm, GroupNorm, MHA/MQA/GQA, RoPE, SwiGLU |
| Architecture & Scaling | Efficient training and large-scale model design | FlashAttention, MLA, DeepSeekMoE, TP/PP/EP/FSDP, Gradient Checkpointing, Mixed Precision Training |
| Adaptation & Alignment | Task adaptation and preference learning | LoRA family, SFT, RLHF, DPO, PPO, GRPO |
| Agent Systems | Retrieval, memory, tool use, API interfaces, and task orchestration | Agent Basics, Memory Systems, RAG Systems, OpenAI API Interfaces |
| Inference & Serving | Latency, memory, and deployment efficiency | Speculative Decoding, Continuous Batching, Quantization, TensorRT-LLM, Hallucination Mitigation |
| Model Zoo | Language models, vision-language models, and robotics systems | DeepSeek, Gemma, CLIP, SigLIP, SmolVLM, robot policies, embodied datasets, and real-time policy inference |

## 1. Foundations
- Math and numerical basics: [SVD](./docs/Math/SVD.md), [dtypes](./docs/Math/dtypes.md), [Memory Estimation](./docs/Math/Memory_Estimation.md), [AdamW](./docs/Optimizer/AdamW.md), [Learning Rate Schedulers](./docs/Scheduler/Scheduler_Basics.md)
- Activation basics: [Sigmoid](./docs/Activation_Layers/Sigmoid.md), [GELU](./docs/Activation_Layers/GELU.md)
- Normalization basics: [LayerNorm](./docs/Norm/LayerNorm.md), [RMSNorm](./docs/Norm/RMSNorm.md), [BatchNorm](./docs/Norm/BatchNorm.md), [GroupNorm](./docs/Norm/GroupNorm.md)
- Attention mechanisms: [SVD + Attention](./docs/Attention_Machanisms/SVD_Attention.md), [MHA](./docs/Attention_Machanisms/MHA.md), [MQA](./docs/Attention_Machanisms/MQA.md), [GQA](./docs/Attention_Machanisms/GQA.md)
- Position and FFN blocks: [Sinusoidal Position Embedding](./docs/Position_Embeding/Sinusoidal_Position_Embedding.md), [RoPE](./docs/Position_Embeding/RoPE.md), [SwiGLU](./docs/Activation_Layers/SwiGLU.md)

## 2. Architecture & Scaling
- Efficient attention: [FlashAttention](./docs/Attention_Machanisms/FlashAttention.md), [MLA](./docs/Attention_Machanisms/MLA.md)
- Sparse architecture: [DeepSeekMoE](./docs/MoE/DeepSeekMoE.md)
- Training memory and numerics: [Gradient Checkpointing](./docs/Training_Optimization/Gradient_Checkpointing.md), [Mixed Precision Training](./docs/Training_Optimization/Mixed_Precision_Training.md)
- Distributed training: [TP](./docs/Parallelism/TP.md), [PP](./docs/Parallelism/PP.md), [EP](./docs/Parallelism/EP.md), [FSDP](./docs/Parallelism/FSDP.md)

## 3. Adaptation & Alignment
- PEFT: [LoRA](./docs/PEFT/LoRA.md), [QLoRA](./docs/PEFT/QLoRA.md), [DoRA](./docs/PEFT/DoRA.md), [Specialized LoRA Variants](./docs/PEFT/Specialized_LoRA.md)
- Supervised and preference alignment: [SFT](./docs/Preference_Alignment/SFT.md), [RLHF](./docs/Preference_Alignment/RLHF.md), [DPO](./docs/Preference_Alignment/DPO.md), [PPO](./docs/Preference_Alignment/PPO.md), [GRPO](./docs/Preference_Alignment/GRPO.md)

## 4. Inference & Serving
- Decoding acceleration: [Speculative Decoding (Medusa/Lookahead)](./docs/Inference_Optimization/speculative_decoding.md)
- Serving systems: [Continuous Batching & PagedAttention](./docs/Inference_Optimization/continuous_batching.md), [TensorRT-LLM & Multi-LoRA Serving](./docs/Inference_Optimization/tensorrt_multilora.md)
- Compression and reliability: [Post-Training Quantization (AWQ/GPTQ)](./docs/Inference_Optimization/quantization_inference.md), [Hallucination Mitigation at Inference](./docs/Inference_Optimization/hallucination_mitigation.md)

## 5. Agent Systems
- Core concepts: [Agent Systems Basics](./docs/Agent_Systems/Agent_Basics.md)
- Memory design: [Memory Systems for Agents](./docs/Agent_Systems/Memory_Systems.md)
- Retrieval grounding: [RAG Systems](./docs/Agent_Systems/RAG_Systems.md)
- Reusable task methods: [Skill Systems](./docs/Agent_Systems/Skill_Systems.md)
- Tool architecture: [Tool Registry and Function Calling](./docs/Agent_Systems/Tool_Registry_and_Function_Calling.md)
- API interface format: [OpenAI API Interface Format](./docs/Agent_Systems/OpenAI_API_Interface_Format.md)
- Protocol layer: [Model Context Protocol (MCP)](./docs/Agent_Systems/MCP_Protocol.md)

## 6. Model Zoo
- Language models: [DeepSeek-V2](./docs/Model_Zoo/Language_Models/DeepSeek_V2.md), [DeepSeek-V3](./docs/Model_Zoo/Language_Models/DeepSeek_V3.md), [DeepSeek-V3.2](./docs/Model_Zoo/Language_Models/DeepSeek_V32.md), [DeepSeek-R1](./docs/Model_Zoo/Language_Models/DeepSeek_R1.md)
- Vision-language models: [CLIP](./docs/Model_Zoo/Vision_Language_Models/CLIP.md), [SigLIP](./docs/Model_Zoo/Vision_Language_Models/SigLIP.md), [SmolVLM](./docs/Model_Zoo/Vision_Language_Models/SmolVLM.md), [Gemma 3](./docs/Model_Zoo/Vision_Language_Models/Gemma_3.md), [Gemma 4](./docs/Model_Zoo/Vision_Language_Models/Gemma_4.md), [DeepSeek-VL](./docs/Model_Zoo/Vision_Language_Models/DeepSeek_VL.md), [DeepSeek-VL2](./docs/Model_Zoo/Vision_Language_Models/DeepSeek_VL2.md)
- Robotics datasets: [LIBERO](./docs/Model_Zoo/Robotics/Datasets/LIBERO.md), [Open X-Embodiment](./docs/Model_Zoo/Robotics/Datasets/Open_X_Embodiment.md)
- Robot policies: [ACT / ALOHA](./docs/Model_Zoo/Robotics/Policies/ACT.md), [Diffusion Policy](./docs/Model_Zoo/Robotics/Policies/Diffusion_Policy.md), [Octo](./docs/Model_Zoo/Robotics/Policies/Octo.md), [RT-1](./docs/Model_Zoo/Robotics/Policies/RT_1.md), [RT-2](./docs/Model_Zoo/Robotics/Policies/RT_2.md), [OpenVLA](./docs/Model_Zoo/Robotics/Policies/OpenVLA.md), [pi0](./docs/Model_Zoo/Robotics/Policies/Pi_0.md), [pi0.5](./docs/Model_Zoo/Robotics/Policies/Pi_0_5.md), [pi0-FAST](./docs/Model_Zoo/Robotics/Policies/Pi_0_FAST.md), [Hi Robot](./docs/Model_Zoo/Robotics/Policies/Hi_Robot.md), [SmolVLA](./docs/Model_Zoo/Robotics/Policies/SmolVLA.md)
- Robotics inference: [Real-Time Chunking](./docs/Model_Zoo/Robotics/Inference/RTC.md)

---

# File Structure
```text
.
|-- assets/
|   `-- ...
`-- docs/
    |-- Agent_Systems/
    |   |-- Agent_Basics.md
    |   |-- MCP_Protocol.md
    |   |-- Memory_Systems.md
    |   |-- OpenAI_API_Interface_Format.md
    |   |-- RAG_Systems.md
    |   |-- Skill_Systems.md
    |   `-- Tool_Registry_and_Function_Calling.md
    |-- Activation_Layers/
    |   |-- GELU.md
    |   |-- Sigmoid.md
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
    |-- Math/
    |   |-- Memory_Estimation.md
    |   |-- SVD.md
    |   `-- dtypes.md
    |-- Model_Zoo/
    |   |-- Language_Models/
    |   |   |-- DeepSeek_R1.md
    |   |   |-- DeepSeek_V2.md
    |   |   |-- DeepSeek_V3.md
    |   |   `-- DeepSeek_V32.md
    |   |-- Robotics/
    |   |   |-- Datasets/
    |   |   |   |-- LIBERO.md
    |   |   |   `-- Open_X_Embodiment.md
    |   |   |-- Inference/
    |   |   |   `-- RTC.md
    |   |   `-- Policies/
    |   |       |-- ACT.md
    |   |       |-- Diffusion_Policy.md
    |   |       |-- Hi_Robot.md
    |   |       |-- Octo.md
    |   |       |-- OpenVLA.md
    |   |       |-- Pi_0.md
    |   |       |-- Pi_0_5.md
    |   |       |-- Pi_0_FAST.md
    |   |       |-- RT_1.md
    |   |       |-- RT_2.md
    |   |       `-- SmolVLA.md
    |   `-- Vision_Language_Models/
    |       |-- CLIP.md
    |       |-- DeepSeek_VL.md
    |       |-- DeepSeek_VL2.md
    |       |-- Gemma_3.md
    |       |-- Gemma_4.md
    |       |-- SigLIP.md
    |       `-- SmolVLM.md
    |-- MoE/
    |   `-- DeepSeekMoE.md
    |-- Norm/
    |   |-- BatchNorm.md
    |   |-- GroupNorm.md
    |   |-- RMSNorm.md
    |   `-- LayerNorm.md
    |-- Optimizer/
    |   `-- AdamW.md
    |-- PEFT/
    |   |-- DoRA.md
    |   |-- LoRA.md
    |   |-- QLoRA.md
    |   `-- Specialized_LoRA.md
    |-- Parallelism/
    |   |-- EP.md
    |   |-- FSDP.md
    |   |-- PP.md
    |   `-- TP.md
    |-- Position_Embeding/
    |   |-- RoPE.md
    |   `-- Sinusoidal_Position_Embedding.md
    |-- Preference_Alignment/
    |   |-- DPO.md
    |   |-- GRPO.md
    |   |-- PPO.md
    |   |-- RLHF.md
    |   `-- SFT.md
    |-- Scheduler/
    |   |-- Cyclical_and_Restart.md
    |   |-- LLM_Training_Recipes.md
    |   |-- Metric_Adaptive.md
    |   |-- Scheduler_Basics.md
    |   `-- Warmup_and_Decay.md
    `-- Training_Optimization/
    |   |-- Gradient_Checkpointing.md
    |   `-- Mixed_Precision_Training.md
```

---

# Learning Resource Recommendation
- [Datawhale / happy-llm](https://github.com/datawhalechina/happy-llm)
- [Datawhale / hello-agents](https://github.com/datawhalechina/Hello-Agents)
- [Datawhale / all-in-rag](https://github.com/datawhalechina/all-in-rag)
- [Hugging Face / LeRobot](https://huggingface.co/docs/lerobot/main/en/index)
- [Physical Intelligence / Home](https://www.pi.website/)
