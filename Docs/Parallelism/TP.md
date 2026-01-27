# Model Parallelism: Sharding Strategies

Shards within a layer (e.g., linear weights along dim) across GPUs

## For Linear Layer: `Y = XA + b`

Weights are sharded across GPUs to fit huge models.

### Column-parallel
Shards output dimension

* **Process:** Share `X`, Shard `A` through output dimension: `A = [A₁, ..., Aₚ]`
* **Each rank produces:** Partial intermediate `Yᵢ = XAᵢ`
* **Combine:** Concatenate intermediate through output dimension

```
X (shared) → [A₁ | A₂ | ... | Aₚ] → [Y₁ | Y₂ | ... | Yₚ] → Concatenate → Full Y
```

### Row-parallel
Shards input dimension

* **Process:** Shard both `X` and `A`:

```
       ┌     ┐
       │ A₁  │
       │ ·   │
A =    │ ·   │      X = [X₁, ..., Xₚ]
       │ ·   │
       │ Aₚ  │
       └     ┘
```

* **Each rank computes:** Partial output `Yᵢ = XᵢAᵢ`
* **Combine:** Reduce partial output (sum parts via All-reduce)

```
X₁ → A₁ → Y₁
X₂ → A₂ → Y₂  → All-reduce(sum) → Full Y
...
Xₚ → Aₚ → Yₚ
```