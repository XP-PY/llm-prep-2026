# DeepSeekMoE
![Model_Architecture](../Resource/pics/DeepSeek-V2.png)

## Overview & Role

* **What is DeepSeekMoE?** Custom Mixture-of-Experts (MoE) architecture in DeepSeek-V2 (paper Section 2.2, 2024) replacing dense FFN for economical training (sparse compute) and strong performance (high capacity/specialization).
    * Total params: Massive (e.g., 236B in V2)
    * Active per token: Small (~21B) → cheap per-token compute
* **Core Idea:** Many fine-grained routed experts (top-K per token, sparse) + few shared experts (always active, dense backbone)
    * Replaces single large FFN → sparse activation for efficiency
* **Real-World Impact:** 42.5% lower training cost vs. dense DeepSeek 67B, stronger performance (top open-source MoE)
* **Historical Context:** Builds on GShard (2021 top-2), Switch (2021 top-1), Mixtral (2023 8x7B SwiGLU MoE). DeepSeek adds fine-grained + shared isolation for better specialization/stability

## Key Components & How It Works

### Experts Structure (Uniform SwiGLU MLP)

* Both shared and routed use same DeepseekV2MLP ([SwiGLU gating](../Activation_Layers/SwiGLU.md))
* Per expert: gate_up_proj (merged fused) + silu(gate) * up + down_proj
* Pseudo code (per expert forward):

```python
def expert_forward(x):  # x: token hidden
    gate_up = gate_up_proj(x)   # Fused [gate | up]
    gate, up = split(gate_up)
    gated = silu(gate) * up     # Dynamic selection
    return down_proj(gated)     # Back to hidden
```

### Shared Experts

* **Quantity:** Few (config.n_shared_experts, e.g., 2–4)
* **Activation:** Always (dense)
* **Intermediate:** Scaled (moe_inter * n_shared) if multiple
* **Role:** Stability/backbone (common knowledge, prevent routing collapse)

### Routed Experts

* **Quantity:** Many fine-grained (config.n_routed_experts, e.g., 256+)
* **Activation:** Top-K per token (config.num_experts_per_tok, e.g., 6–8)
* **Role:** Specialization (narrow skills, e.g., math/code/Chinese)

### Routing (Gate)

* Lightweight linear: hidden → n_routed_experts logits
* Top-K selection (grouped for speed: n_group/topk_group)
* Pseudo code:

```python
logits = gate(hidden_tokens)  # [tokens, n_experts]
topk_indices, topk_weights = grouped_topk(logits, k=top_k, groups=n_group)
routed_out = sum(weight_i * expert_i(token) for i in topk)
```

### Full MoE Output

```
final = routed_out * scaling_factor + shared_out
```

* Scaling: For FP16 stability (routed damped, shared boosted)

### Process Flow (vLLM Pseudo Overview)

```python
hidden = hidden_states.flatten()
router_logits = gate(hidden)  # Scores

# Fused kernel: Dispatch top-K tokens to local experts (EP all-to-all)
routed_out = fused_experts(hidden, router_logits)  # Sparse compute

shared_out = shared_mlp(hidden) if shared else 0  # Dense
final = routed_out * scaling + shared_out
all_reduce/gather if TP/seq_parallel
return reshape(final)
```

### Parallelism & Stability Tricks

* **Expert Parallel (EP):** Shard routed (physical = logical + redundant for EPLB balance)
* **Tensor Parallel (TP):** Within-expert sharding (merged column for gate_up, row for down)
* **Expert Parallel Load Balancing (EPLB):** Redundant duplicates → route to less-loaded copies (smooth comm/load)
* **Grouped Top-K:** Divide experts into groups → faster local top-K
* **Other:** Renormalize probs, e_score_bias, token-dropping (paper for balance)

## Benefits, Trade-offs & Comparisons

| Aspect | Dense FFN (e.g., DeepSeek 67B) | DeepSeekMoE (V2) |
|:--------:|:-------------------------------:|:------------------:|
| Params Total | Fixed (67B) | Massive (236B) |
| Active per Token | All | ~21B (sparse top-K + shared) |
| Training Cost | High | 42.5% lower (sparse) |
| Specialization | General | High (fine-grained routed) |
| Stability | Good | Enhanced (shared + tricks) |
| Trade-off | Simpler | Comm overhead (EP mitigated) |

* **Why stronger?** Sparse capacity + gating (SwiGLU) → better scaling/specialization
* **Edge cases:** Routing collapse (mitigated by shared/aux), FP16 overflow (scaling fix)