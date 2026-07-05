# Gradient Checkpointing

## Convenient Links

* [PyTorch checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html)
* [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
* [FSDP note in this repo](../Parallelism/FSDP.md)
* [FlashAttention note in this repo](../Attention_Machanisms/FlashAttention.md)
* [Memory Estimation note in this repo](../Math/Memory_Estimation.md)
* [Mixed Precision Training note in this repo](./Mixed_Precision_Training.md)

## 1. One-Sentence Summary

Gradient checkpointing is a **training-time activation memory optimization** that saves only selected intermediate tensors during the forward pass, then recomputes missing activations during backward, trading extra compute for lower GPU memory.

## 2. Why Gradient Checkpointing Matters

During LLM training, GPU memory is spent on several things:

* parameters
* gradients
* optimizer states
* activations
* temporary buffers
* communication buffers in distributed training

For long-context Transformer training, **activations** can become one of the largest memory terms because they scale with:

```text
batch size x sequence length x hidden size x number of layers
```

This is why simply sharding parameters with FSDP is not always enough. FSDP reduces model-state memory, but activations are still produced by each rank's local microbatch.

Gradient checkpointing attacks this activation-memory problem.

## 3. Core Idea

Normal training saves many forward-pass intermediates because backward needs them.

```text
forward:
  x -> layer1 -> layer2 -> layer3 -> loss
  save activations for backward

backward:
  reuse saved activations
  compute gradients
```

Gradient checkpointing saves fewer activations:

```text
forward:
  x -> checkpointed block -> output
  save only block input / output

backward:
  recompute block forward
  recover missing activations
  compute gradients
```

The tradeoff is:

| Resource | Effect |
| :------- | :----- |
| GPU memory | lower |
| Training compute | higher |
| Wall-clock time | usually slower |
| Numerical result | intended to be equivalent, modulo dropout/RNG details |

## 4. Activation Memory Problem

For a Transformer block, backward often needs tensors from:

* Q/K/V projections
* attention probabilities or attention intermediates
* MLP hidden activations
* residual streams
* normalization inputs

Without checkpointing, many of these are kept until backward.

If the model has `L` layers, the activation memory roughly grows with:

$$
O(L)
$$

because the training graph stores intermediates for many layers.

Checkpointing reduces how many layer activations are retained at once.

## 5. Segment Checkpointing Mental Model

Suppose a model has `L` layers split into `S` segments.

```text
layers:
[1 2 3 4] [5 6 7 8] [9 10 11 12]
 segment 1 segment 2 segment 3
```

Instead of saving every layer's internal activations, the system saves segment boundaries.

During backward through segment 3:

```text
recompute layers 9-12
use recomputed activations
backprop through segment 3
discard segment 3 intermediates
```

Then it repeats for segment 2 and segment 1.

The more aggressively you checkpoint, the less activation memory you keep, but the more forward compute you repeat.

## 6. Compute Tradeoff

A plain training step already does:

```text
1 forward pass
1 backward pass
```

Checkpointing adds partial extra forward computation during backward.

In the worst simple mental model, checkpointed layers may run their forward computation twice:

```text
first time: original forward
second time: recomputation during backward
```

So training can become noticeably slower. In practice, the slowdown depends on:

* how many layers are checkpointed
* how expensive attention is relative to MLP
* whether FlashAttention or fused kernels are used
* sequence length
* communication overlap in distributed training
* GPU utilization before checkpointing

The memory savings are often worth it if checkpointing lets you use:

* a larger batch size
* a longer context length
* a larger model
* fewer GPUs

## 7. Transformer-Level Usage

In LLMs, checkpointing is usually applied at the Transformer-block level:

```text
embedding
-> block 1  checkpointed
-> block 2  checkpointed
-> ...
-> block N  checkpointed
-> lm head
```

This is common because Transformer blocks are natural recomputation units:

* each block has a clear input and output
* memory-heavy internals can be dropped
* recomputation cost is predictable
* implementation is simple in PyTorch and Hugging Face models

Some systems checkpoint every block. Others checkpoint every few blocks or only the most memory-heavy blocks.

## 8. PyTorch Example

Minimal PyTorch pattern:

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)
```

For modern PyTorch, `use_reentrant=False` is often the recommended path because it supports more autograd features and avoids several older reentrant-checkpointing limitations.

In Hugging Face Transformers, the common switch is:

```python
model.gradient_checkpointing_enable()
```

During training, it is also common to disable generation-time cache:

```python
model.config.use_cache = False
```

because KV caching is for autoregressive inference, not standard full-sequence training.

## 9. RNG and Dropout

Checkpointing recomputes forward operations during backward.

That creates a subtle issue for random operations such as:

* dropout
* stochastic depth
* random masking

If recomputation uses different random masks, gradients will not match the original forward pass.

Checkpointing systems usually preserve RNG state so recomputation sees the same randomness. This has some overhead, but it keeps the training objective consistent.

Practical rule:

> If a checkpointed block contains dropout, use the framework's normal checkpoint API rather than hand-rolling recomputation.

## 10. Interaction with FlashAttention

FlashAttention already avoids materializing the full attention matrix:

$$
S = QK^T
$$

and therefore reduces attention activation memory.

Gradient checkpointing is still useful because a Transformer block has more than the attention matrix:

* Q/K/V inputs and projections
* MLP activations
* normalization inputs
* residual streams

A useful mental model:

```text
FlashAttention:
  reduces attention-specific memory

Gradient checkpointing:
  reduces broader layer activation memory
```

They are complementary, though the incremental gain from checkpointing may be smaller when attention memory has already been optimized.

## 11. Interaction with FSDP and ZeRO

FSDP and ZeRO-style methods reduce model-state memory:

* parameters
* gradients
* optimizer states

Gradient checkpointing reduces activation memory.

So they solve different parts of the memory budget:

| Method | Mainly saves |
| :----- | :----------- |
| FSDP / ZeRO | model states |
| Gradient checkpointing | activations |
| Mixed precision | model states, activations, bandwidth |
| FlashAttention | attention intermediates |

This is why large-model training recipes often combine all of them.

## 12. When to Use It

Use gradient checkpointing when:

* you hit out-of-memory during training
* sequence length is large
* microbatch size is too small for stable training
* FSDP reduces parameters but activations still dominate
* you want to fit a larger model on fixed hardware

Be more cautious when:

* training is already compute-bound and slow
* you have enough memory without it
* your model has unusual random operations inside checkpointed regions
* you are debugging numerics and want the simplest execution path

## 13. Common Failure Modes

### 13.1 Forgetting to Disable `use_cache`

For decoder-only LLM training, generation cache should usually be off:

```python
model.config.use_cache = False
```

If left on, it may increase memory or conflict with gradient checkpointing in some model implementations.

### 13.2 Checkpointing Too Fine-Grained

Checkpointing tiny operations can add overhead without meaningful memory savings.

Prefer natural blocks:

```text
Transformer layer
attention + MLP block
several-layer segment
```

### 13.3 Assuming It Saves Parameter Memory

Gradient checkpointing does not shard or quantize weights.

If parameters or optimizer states dominate memory, use:

* FSDP
* ZeRO
* optimizer offload
* LoRA / QLoRA
* lower precision

### 13.4 Surprise Slowdown

A slowdown is expected. The question is whether the memory savings let you recover throughput with larger batch size or longer sequence length.

## 14. Practical Recipe

For LLM fine-tuning:

```text
start with BF16 mixed precision
enable FlashAttention if supported
enable gradient checkpointing
set use_cache = False
increase sequence length or microbatch until near memory limit
use gradient accumulation to reach target global batch
```

For full pretraining:

```text
combine FSDP / ZeRO with activation checkpointing
checkpoint complete Transformer blocks
measure memory and tokens/sec
tune microbatch size and checkpoint granularity
```

Do not optimize only for minimum memory. Optimize for:

```text
tokens per second at the target global batch and sequence length
```

## 15. Mental Model

The shortest way to remember gradient checkpointing is:

```text
Do not store every activation.
Store a few boundaries.
Recompute missing forward activations during backward.
Spend more compute to buy more memory.
```

It is one of the most practical knobs for making large Transformer training fit on limited GPU memory.
