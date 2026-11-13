# Fully Sharded Data Parallel (FSDP)

## Convenient Links
* [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
* [PyTorch FSDP API](https://docs.pytorch.org/docs/stable/fsdp.html)
* [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)

## 1. Why this note exists

When a model is too large to fit on one GPU, the first naive idea is often:

* reduce batch size
* use gradient checkpointing
* add more GPUs with DDP

But **DDP** still replicates the full model on every rank, so it does **not** solve the fundamental parameter-memory problem.

That is why PyTorch provides **FSDP**.

The clean mental model is:

> FSDP shards **parameters, gradients, and optimizer states** across data-parallel ranks, and only materializes full parameters temporarily around computation.

This is why FSDP is often introduced as the PyTorch-native path for:

* training models that do not fit on one GPU
* scaling large dense Transformers
* reducing optimizer-state memory compared with pure DDP

As of the current official PyTorch documentation, the recommended frontend is **FSDP2** via:

```python
torch.distributed.fsdp.fully_shard
```

while the older wrapper class:

```python
torch.distributed.fsdp.FullyShardedDataParallel
```

is still common in existing codebases but is treated as the **legacy / deprecated FSDP1 path** in the official tutorials.

---

## 2. One-Sentence Summary

FSDP is a **sharded data-parallel training method** that keeps only a shard of model states on each rank outside computation, then uses **all-gather** and **reduce-scatter** to reconstruct and update parameters during forward and backward.

---

## 3. DDP vs FSDP

Suppose the model parameters are:

$$
\theta
$$

and there are `P` data-parallel ranks.

### DDP

In DDP, every rank stores a full copy:

$$
\theta^{(1)} = \theta^{(2)} = \cdots = \theta^{(P)} = \theta
$$

Each rank computes local gradients:

$$
g_r = \nabla_\theta \mathcal{L}_r
$$

and gradients are synchronized with an **all-reduce**:

$$
g = \frac{1}{P}\sum_{r=1}^{P} g_r
$$

This is simple and fast when the model fits, but memory is expensive because:

* parameters are replicated
* gradients are replicated
* optimizer states are replicated

### FSDP

In FSDP, the parameters are partitioned:

$$
\theta = [\theta_1, \theta_2, \dots, \theta_P]
$$

where rank `r` keeps only shard $\theta_r$ outside computation.

The same sharding idea applies to:

* gradients
* optimizer states

So FSDP trades:

* **lower memory**

for:

* **more communication**
* **more complicated runtime behavior**

---

## 4. What Exactly Is Sharded?

FSDP mainly targets three large memory components:

1. **Parameters**
2. **Gradients**
3. **Optimizer states**

If the world size is `P`, then outside computation each rank ideally stores only about:

$$
\frac{1}{P}
$$

of those model states.

This is the key difference from DDP.

However, FSDP does **not** magically eliminate all memory:

* activations are still needed
* full parameters must be materialized transiently for computation
* communication buffers still exist

So FSDP is a large memory win, but not a complete memory solution by itself.

---

## 5. Forward / Backward Workflow

The best way to understand FSDP is to follow one FSDP unit through forward and backward.

### 5.1 Outside Compute

Each rank stores only its local parameter shard:

$$
\theta_r
$$

### 5.2 Before Forward

For the current FSDP unit, ranks perform **all-gather** to reconstruct the full parameters:

$$
\theta = \operatorname{AllGather}(\theta_1, \theta_2, \dots, \theta_P)
$$

### 5.3 Forward Compute

The wrapped module computes normally with full parameters:

$$
y = f(x; \theta)
$$

### 5.4 After Forward

Depending on configuration, the full parameters may be discarded and only local shards kept again.

That is the usual memory-saving path.

### 5.5 Before / During Backward

Backward needs parameter values too, so FSDP may unshard again for the corresponding module.

### 5.6 Gradient Synchronization

Instead of DDP's gradient all-reduce, FSDP uses **reduce-scatter**:

$$
g_r
=
\operatorname{ReduceScatter}\left(
\nabla_\theta \mathcal{L}_1,
\nabla_\theta \mathcal{L}_2,
\dots,
\nabla_\theta \mathcal{L}_P
\right)
$$

So each rank keeps only the gradient shard corresponding to its parameter shard.

### 5.7 Optimizer Step

The optimizer updates only the local shard:

$$
\theta_r \leftarrow \theta_r - \eta g_r
$$

and its optimizer states are also sharded locally.

---

## 6. Why People Say FSDP Is "All-Reduce Decomposed"

A useful viewpoint is:

> DDP's gradient all-reduce can be thought of as a reduce-scatter plus an all-gather.

DDP conceptually keeps everything replicated, so an all-reduce is natural.

FSDP keeps model states sharded, so it uses:

* **all-gather** to reconstruct parameters before compute
* **reduce-scatter** to keep gradients sharded after backward

This is why FSDP is often described as a sharded form of data parallelism rather than a completely different training paradigm.

---

## 7. FSDP Unit and Wrap Granularity

This point is extremely important in practice.

FSDP does not have to wrap the whole model as one monolithic block.
Instead, you often define several **FSDP units**.

Why does this matter?

Because FSDP unshards **per wrapped unit**.

If you wrap too coarsely:

* full-parameter materialization becomes large
* peak memory rises

If you wrap too finely:

* communication frequency increases
* runtime overhead rises

So FSDP performance depends heavily on choosing a reasonable wrap boundary, often:

* per Transformer block
* per decoder layer
* per large repeated module

This is why auto-wrap policies and manual submodule wrapping matter a lot.

---

## 8. FSDP1 vs FSDP2

This distinction matters because you will see both in the wild.

### FSDP1

Legacy wrapper class:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
```

Characteristics:

* flat-parameter style internal representation
* many older tutorials and repos use it
* rich wrapper configuration API
* still important for reading existing code

### FSDP2

Current recommended frontend:

```python
from torch.distributed.fsdp import fully_shard
```

Characteristics:

* parameter sharding represented via **DTensor**
* more explicit submodule sharding style
* simpler mental model for current PyTorch users
* official tutorial path now centers on this API

### Practical Recommendation

For **new PyTorch code**, start with **FSDP2** unless you are constrained by an existing FSDP1 codebase.

For **existing projects**, you still need to recognize FSDP1 because many real training stacks have not fully migrated yet.

---

## 9. Current PyTorch Decision Rule

The official PyTorch distributed overview gives a simple guideline:

* use **DDP** if the model fits on one GPU
* use **FSDP2** if the model does not fit on one GPU
* consider **TP / PP** if FSDP alone reaches scaling limits

That is a very good first-pass rule in practice.

You can remember it like this:

* **DDP**: replicate model, split data
* **FSDP**: shard model states, still data parallel
* **TP / PP**: if one layer or one stage is still too large or too slow

---

## 10. FSDP Sharding Styles

You will commonly see the following FSDP1 strategies:

* `FULL_SHARD`
* `SHARD_GRAD_OP`
* `HYBRID_SHARD`

### `FULL_SHARD`

The strongest memory-saving style.

Interpretation:

* parameters are sharded outside computation
* gradients are sharded
* optimizer states are sharded
* parameters are usually resharded after forward

### `SHARD_GRAD_OP`

A slightly less aggressive variant.

Interpretation:

* gradients and optimizer states are sharded
* parameters are not reshared immediately after forward

This can save some communication at the cost of higher memory.

### `HYBRID_SHARD`

Common multi-node idea:

* shard within a node
* replicate across nodes

This is often useful because:

* intra-node links are much faster
* expensive all-gathers / reduce-scatters can stay local to a node

### FSDP2 Mapping

In the current official FSDP2 migration guidance:

* `FULL_SHARD` roughly maps to `reshard_after_forward=True`
* `SHARD_GRAD_OP` roughly maps to `reshard_after_forward=False`
* `HYBRID_SHARD` uses a **2D device mesh**

---

## 11. Minimal FSDP2 Training Example

Below is a small but realistic FSDP2 skeleton.

Launch:

```bash
torchrun --nproc_per_node=4 train_fsdp2.py
```

Script:

```python
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard


class MLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ToyTransformer(nn.Module):
    def __init__(self, dim: int = 1024, num_layers: int = 8) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([MLPBlock(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)


def main() -> None:
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = ToyTransformer().to(device)

    # Wrap repeated blocks first, then wrap the root module.
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(10):
        x = torch.randn(8, 128, 1024, device=device)
        target = torch.randn(8, 128, 1024, device=device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0:
            print(f"step={step} loss={loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### Why this example is written this way

There are several details worth noticing:

1. `torchrun` launches one process per GPU
2. `torch.cuda.set_device(local_rank)` must happen early
3. `optimizer` is created **after sharding**
4. repeated blocks are wrapped first, then the root model

That last point is exactly how current PyTorch FSDP2 examples are structured.

---

## 12. FSDP2 Mixed Precision Example

FSDP2 has its own mixed precision policy.

A common and practical choice is:

* use `bfloat16` for forward / backward parameters
* use `float32` for gradient reduction

Example:

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

model = ToyTransformer()
for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)
fully_shard(model, mp_policy=mp_policy)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

Why this is useful:

* `param_dtype=torch.bfloat16` reduces compute and memory pressure
* `reduce_dtype=torch.float32` keeps gradient communication numerically safer

This is often a very good default on modern GPUs that support `bfloat16`.

---

## 13. FSDP2 Meta Initialization Example

For large models, even constructing the full model on GPU first may OOM.

That is why current PyTorch FSDP2 docs also show a **meta-device initialization** pattern.

Example:

```python
import torch
from torch.distributed.fsdp import fully_shard

with torch.device("meta"):
model = ToyTransformer(dim=4096, num_layers=32)

for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# Materialize only after sharding.
model.to_empty(device="cuda")
for module in model.modules():
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
```

The idea is:

* create the parameter structure first on `meta`
* shard it
* materialize only after the sharding plan is already in place

This can be very important for large models.

---

## 14. FSDP2 Checkpointing Example

Current PyTorch documentation recommends using **Distributed Checkpoint (DCP)** style APIs for state dict handling with FSDP2.

Example:

```python
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

# Save a full CPU-offloaded state dict.
state_dict = get_model_state_dict(
    model,
    options=StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    ),
)

if dist.get_rank() == 0:
    torch.save(state_dict, "model_state_dict.pt")


# Later: load from rank 0 and broadcast to other ranks.
if dist.get_rank() == 0:
    loaded = torch.load("model_state_dict.pt", map_location="cpu")
else:
    loaded = None

set_model_state_dict(
    model=model,
    model_state_dict=loaded,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    ),
)
```

The key idea is:

* rank 0 can hold the full CPU checkpoint
* then the state is broadcast into the sharded model structure

This is cleaner than trying to manually rebuild full weights on every rank.

---

## 15. Legacy FSDP1 Example

You still need to recognize this style because many existing repos use it.

Launch:

```bash
torchrun --nproc_per_node=4 train_fsdp1.py
```

Script:

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def main() -> None:
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = ToyTransformer().cuda()
    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # training loop ...
```

### Why keep this section?

Because in practice you will still read code that uses:

* `FullyShardedDataParallel`
* `auto_wrap_policy`
* `ShardingStrategy.FULL_SHARD`
* `sync_module_states`
* `use_orig_params=True`

So even if you write new code with FSDP2, you should still be able to read FSDP1.

---

## 16. FSDP vs ZeRO

FSDP is often discussed alongside DeepSpeed ZeRO.

The clean relationship is:

* both try to avoid full replication of model states
* both shard parameters / gradients / optimizer states in some form
* FSDP is the **PyTorch-native** implementation path

In many learning notes, you can roughly think of FSDP as being conceptually close to **ZeRO Stage 3** style sharding, though the concrete implementation details and APIs differ.

---

## 17. FSDP vs TP / PP

This distinction matters a lot.

### FSDP

* still a **data-parallel** style method
* each rank runs the same model structure
* model states are sharded across ranks

### TP

* splits one layer across GPUs
* useful when one dense layer is too large or too expensive

### PP

* splits model depth across stages
* useful when the model stack itself is too deep / too large for one rank group

So a common progression is:

1. try DDP if the model fits
2. move to FSDP if the model does not fit
3. add TP / PP if FSDP alone is not enough

Large production training systems often combine them:

```text
world size = DP(or FSDP data-parallel group) x TP x PP
```

and for MoE:

```text
world size = FSDP x TP x PP x EP
```

---

## 18. Common Pitfalls

### 18.1 Building the optimizer too early

Do **not** construct the optimizer before sharding.

Wrong:

```python
model = ToyTransformer()
optimizer = AdamW(model.parameters(), lr=1e-4)
fully_shard(model)
```

Correct:

```python
model = ToyTransformer()
fully_shard(model)
optimizer = AdamW(model.parameters(), lr=1e-4)
```

### 18.2 Wrapping too coarsely

If you wrap the whole model as one unit, peak memory may still be too high because one very large parameter set is unsharded together.

### 18.3 Wrapping too finely

Too many tiny FSDP units increase communication overhead and often hurt throughput.

### 18.4 Forgetting rank-to-device mapping

`LOCAL_RANK` and `torch.cuda.set_device(local_rank)` are basic but essential.

### 18.5 Assuming FSDP solves everything

FSDP helps with model-state memory, but:

* activations may still dominate
* very large layers may still require TP
* very large multi-node runs may still need PP / hybrid strategies

### 18.6 Using the wrong gradient clipping API

In current FSDP2 examples, standard:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

works naturally with the DTensor-based parameter view.

In older FSDP1 code, you may instead see:

```python
model.clip_grad_norm_(max_norm)
```

because sharded gradients required FSDP-aware clipping logic.

---

## 19. When FSDP Is a Good Choice

FSDP is especially attractive when:

* the model does not fit on one GPU
* you want to stay close to the PyTorch-native stack
* you mainly need memory reduction rather than exotic model partitioning
* your model is still mostly dense

FSDP is less likely to be sufficient alone when:

* one single layer is too large even transiently
* activation memory dominates
* you need very aggressive multi-dimensional scaling

In those cases, TP / PP or additional activation strategies become more important.

---

## 20. Key Takeaways

1. **FSDP shards parameters, gradients, and optimizer states across data-parallel ranks**
2. **Its core runtime pattern is: all-gather full parameters before compute, reduce-scatter gradients after backward**
3. **The current recommended PyTorch frontend is FSDP2 via `fully_shard`; FSDP1 remains important mainly for reading older code**
4. **Wrap granularity is one of the most important practical tuning decisions**
5. **In PyTorch practice, use DDP if the model fits one GPU, FSDP if it does not, and add TP / PP if FSDP alone reaches its limits**
