# Expert Parallelism (EP) in MoE Models

## Overview & Role

* **What is Expert Parallelism (EP)?** A distributed parallelism strategy for Mixture-of-Experts (MoE) models, sharding routed experts across GPUs.
    * Each GPU (rank) owns a slice of the many experts → tokens dispatched via all-to-all communication.
    * Role: Fits massive fine-grained routed experts in memory, enables scaling [MoE](../MoE/DeepSeekMoE.md) without single-GPU limit.
* **Historical Context:** From GShard (2021, Google MoE with EP), Switch Transformers (2021 top-1 routing), Mixtral (2023 8x7B MoE). [DeepSeek-V2](../Large_Models/DeepSeek_V2.md) refines with fine-grained + EPLB for stability.

## How It Works (Step-by-Step)

### Basic EP

* Routed experts divided evenly: `n_routed_experts // ep_size` per rank.
* <span style="color: blue">Gate/router computes top-K per token</span> → dispatch tokens to expert-owning ranks (all-to-all comm).
* Local compute on owned experts → combine outputs.

### With EPLB (Expert Parallel Load Balancing, DeepSeek Enhancement)

* Add <span style="color: blue">redundant duplicates</span> (`n_redundant_experts`).
* Physical = logical + redundant → dispatch to any copy (choose less-loaded).
* Smooths imbalance (popular experts overloaded).

### Pseudo Code (vLLM DeepseekV2MoE tie-in)

```python
# Init
logical_experts = config.n_routed_experts                # e.g., 256
redundant = eplb_config.num_redundant_experts           # e.g., 32
physical = logical + redundant                          # 288
local_physical = physical // ep_size                    # Per-rank slice (includes duplicates)

# Forward
logits = gate(hidden)                                   # Top-K (grouped for speed)
topk_indices = grouped_topk(logits)                     # Select physical copies (balance load)

# All-to-all: Send tokens to expert owners
dispatched_tokens = all_to_all(topk_indices, hidden)

# Local compute on owned physical experts (SwiGLU MLP)
local_out = local_experts(dispatched_tokens)

# All-to-all back + combine weighted sum
routed_out = all_to_all_back(local_out, weights)
```

### Diagram Description

```
Tokens → Gate/Router → Top-K indices (grouped, to physical copies)

All-to-All Dispatch:
GPU0 (owns experts 0-35) ← tokens for its copies
GPU1 (36-71) ← ...
...
All-to-All Back: Combine local expert outputs → routed sum

+ Shared experts (local/dense) → Final MoE out
```

## Benefits, Trade-offs & Comparisons

### Benefits

* **Memory:** Fits thousands of experts (fine-grained specialization).
* **Compute:** Sparse per token (top-K only).
* **Balance (EPLB):** Reduces hotspots → stable training/inference.

### Trade-offs

* **Comm overhead:** All-to-all expensive if imbalance → EPLB/redundancy mitigates.
* **Complexity:** Dispatch logic, redundancy memory cost.
* **Edge:** Small MoE → unnecessary (TP sufficient).

### Parallelism Comparison

| Aspect | No Parallelism (Single GPU) | [Tensor Parallel (TP)](./TP.md) | Expert Parallel (EP) + EPLB |
|--------|----------------------------|---------------------|----------------------------|
| Sharded | None | Within-layer weights | Whole routed experts |
| Comm Type | None | All-reduce/gather | All-to-all (token dispatch) |
| Memory Win | Limited | Dense params | Many experts (fine-grained) |
| Balance Issue | N/A | Even | Imbalance (EPLB fixes) |
| Typical in V2 | Impossible | Attention/shared | Routed experts |

## Broader Implications

Enables open-source MoE scaling (V2 top-tier with economy). Combined TP+EP standard for large MoE.