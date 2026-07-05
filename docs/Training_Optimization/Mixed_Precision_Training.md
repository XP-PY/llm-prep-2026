# Mixed Precision Training

## Convenient Links

* [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
* [PyTorch CUDA semantics: TF32](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
* [Common dtypes note in this repo](../Math/dtypes.md)
* [Gradient Checkpointing note in this repo](./Gradient_Checkpointing.md)
* [Memory Estimation note in this repo](../Math/Memory_Estimation.md)
* [FSDP note in this repo](../Parallelism/FSDP.md)

## 1. One-Sentence Summary

Mixed precision training uses lower-precision formats such as **BF16** or **FP16** for most heavy tensor operations while keeping sensitive values, accumulations, or optimizer states in higher precision when needed, improving memory use, bandwidth, and GPU throughput.

## 2. Why Mixed Precision Matters

LLM training is limited by:

* GPU memory
* memory bandwidth
* matrix multiplication throughput
* communication bandwidth across GPUs

Using FP32 everywhere is stable, but expensive:

```text
FP32 = 4 bytes per value
FP16 = 2 bytes per value
BF16 = 2 bytes per value
```

Moving from FP32 to BF16/FP16 can roughly halve memory for many tensors and can unlock faster Tensor Core kernels on modern GPUs.

The goal is not to make everything low precision. The goal is:

> Use low precision where it is fast and safe, and keep enough precision where training would otherwise become unstable.

## 3. The Main Dtypes

| Dtype | Bytes | Strength | Main risk |
| :---- | ----: | :------- | :-------- |
| FP32 | `4` | stable baseline | high memory and bandwidth |
| TF32 | storage is FP32, matmul uses reduced precision | fast FP32-like matmul on NVIDIA Ampere+ | lower mantissa precision |
| FP16 | `2` | fast and memory-efficient | narrow exponent range, overflow/underflow |
| BF16 | `2` | FP32-like exponent range, stable training | less mantissa precision than FP16 |
| FP8 | `1` | very high throughput on newer hardware | needs careful scaling and hardware support |

For most modern LLM training:

```text
prefer BF16 if the hardware supports it
otherwise use FP16 with loss scaling
```

## 4. FP16 vs BF16

FP16 and BF16 both use 16 bits, but they allocate those bits differently.

| Format | Exponent bits | Mantissa bits | Practical meaning |
| :----- | ------------: | ------------: | :---------------- |
| FP16 | `5` | `10` | more precision near 1, smaller dynamic range |
| BF16 | `8` | `7` | FP32-like dynamic range, less fine precision |

The dynamic range difference is usually more important for deep learning training.

FP16 can underflow small gradients and overflow large activations. This is why FP16 training often needs loss scaling.

BF16 usually does not need loss scaling because it has the same exponent width as FP32.

## 5. What "Mixed" Means

Mixed precision does not mean every tensor uses the same dtype.

A typical training setup may look like:

| Tensor / operation | Common dtype |
| :----------------- | :----------- |
| Matrix multiplications | BF16 or FP16 |
| Activations | BF16 or FP16 |
| Gradients | BF16/FP16 communication, sometimes FP32 accumulation |
| LayerNorm / RMSNorm | often FP32 or higher-precision internal compute |
| Softmax | often higher-precision internal compute |
| Optimizer states | often FP32 |
| Master weights in FP16 training | often FP32 |

The exact choices depend on framework, model, optimizer, and distributed strategy.

## 6. Autocast

PyTorch AMP uses autocast to choose appropriate dtypes for operations:

```python
import torch

model = model.cuda()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(input_ids)
    loss = outputs.loss
```

Autocast does not blindly cast every operation. It uses rules so that operations known to be numerically sensitive can run in a safer dtype.

For BF16 training on supported GPUs, the loop can often be simple:

```python
optimizer.zero_grad()

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(**batch).loss

loss.backward()
optimizer.step()
```

## 7. Loss Scaling for FP16

FP16 has a narrow exponent range. Small gradients can underflow to zero.

Loss scaling multiplies the loss before backward:

$$
\tilde{L} = sL
$$

This scales gradients:

$$
\nabla_\theta \tilde{L} = s \nabla_\theta L
$$

Before the optimizer step, gradients are unscaled:

$$
\nabla_\theta L = \frac{1}{s}\nabla_\theta \tilde{L}
$$

If overflow is detected, the optimizer step is skipped and the scale is reduced.

PyTorch FP16 pattern:

```python
scaler = torch.cuda.amp.GradScaler()

optimizer.zero_grad()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(**batch).loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

For BF16, loss scaling is usually unnecessary.

## 8. Master Weights and Optimizer States

In classic FP16 training, the model may keep:

* FP16 weights for forward/backward
* FP32 master weights for optimizer updates
* FP32 optimizer states such as Adam moments

Why?

Adam updates can be small. Applying tiny updates directly to FP16 weights can lose precision.

So the optimizer updates FP32 master weights, then casts back to FP16 for compute.

With BF16 training, many modern systems still keep optimizer states in FP32, even if parameters and activations are BF16.

Memory implication:

> Mixed precision reduces many tensors, but optimizer states can still dominate memory unless sharded, offloaded, or quantized.

## 9. Numerically Sensitive Operations

Some operations are more sensitive to reduced precision:

* softmax
* log softmax
* cross entropy
* normalization statistics
* variance computation
* very small residual updates
* long reductions

Frameworks and fused kernels often handle these carefully.

For example, a kernel may:

```text
read BF16 inputs
accumulate in FP32
write BF16 outputs
```

This is still mixed precision.

## 10. TF32

TF32 is different from FP16/BF16.

It is not usually a storage dtype for model weights. It is a compute mode on NVIDIA Ampere+ GPUs where FP32 matrix multiplications use Tensor Cores with reduced mantissa precision.

Practical effect:

```text
FP32 code
-> faster matmul
-> usually similar training behavior
```

TF32 is useful when you want much of FP32's stability but faster matmul performance.

In PyTorch, TF32 behavior can be controlled through matmul precision settings or backend flags.

## 11. FP8 Training

FP8 uses 8-bit floating formats, commonly:

| Format | Common use |
| :----- | :--------- |
| E4M3 | forward activations / weights where precision matters |
| E5M2 | gradients where dynamic range matters |

FP8 training usually needs:

* hardware support, such as NVIDIA Hopper or newer
* scaling factors
* amax tracking
* careful recipe choices
* framework support such as Transformer Engine

FP8 can be powerful, but it is not the default first step for most training notes. BF16 is the practical baseline.

## 12. Memory Impact

Mixed precision saves memory in several places:

| Component | Possible effect |
| :-------- | :-------------- |
| Parameters | FP32 to BF16/FP16 halves storage |
| Activations | often halves activation memory |
| Gradients | can reduce gradient memory and communication bandwidth |
| Optimizer states | often still FP32 unless using sharding or low-bit optimizers |

For Adam, optimizer states are large:

```text
weights
gradients
first moment m
second moment v
```

Even with BF16 weights, Adam states may stay FP32. This is why mixed precision is often combined with:

* FSDP / ZeRO
* 8-bit optimizers
* optimizer offload
* LoRA / QLoRA

## 13. Interaction with Gradient Checkpointing

Mixed precision and gradient checkpointing save different memory:

| Technique | Main target |
| :-------- | :---------- |
| Mixed precision | dtype size, bandwidth, compute throughput |
| Gradient checkpointing | activation storage |

They are commonly used together:

```text
BF16 mixed precision
+ gradient checkpointing
+ FlashAttention
+ FSDP
```

One subtlety: checkpointed recomputation should run under the same autocast behavior as the original forward. Framework checkpoint APIs handle this better than manual recomputation.

## 14. Common Failure Modes

### 14.1 FP16 Overflow or NaNs

Symptoms:

* loss becomes `nan`
* gradients become `inf`
* training diverges after a few steps

Fixes:

* use BF16 if supported
* use GradScaler for FP16
* lower learning rate
* add or reduce gradient clipping depending on the issue
* keep norms/softmax in higher precision

### 14.2 Assuming BF16 Is Always Identical to FP32

BF16 has FP32-like range, but fewer mantissa bits.

It is usually stable for LLM training, but small numerical differences can still affect:

* exact reproducibility
* very small models or losses
* sensitive scientific workloads

### 14.3 Optimizer Memory Still Too High

Mixed precision may not solve memory if Adam states dominate.

Use:

* FSDP / ZeRO
* 8-bit optimizer states
* LoRA / QLoRA
* smaller batch or sequence length

### 14.4 Wrong Dtype for Hardware

Older GPUs may have poor BF16 support. Some consumer GPUs support FP16 much better than BF16.

Always check actual throughput instead of assuming one dtype is fastest.

## 15. Practical Defaults

For modern NVIDIA A100/H100/L40/L4 style training:

```text
use BF16 autocast
keep optimizer states in FP32 unless using a known low-bit optimizer
use FlashAttention when available
use gradient checkpointing for long context
use FSDP / ZeRO for large models
```

For older FP16-oriented GPUs:

```text
use FP16 autocast
use GradScaler
watch for NaNs and inf gradients
consider lower learning rate or stronger gradient clipping
```

For QLoRA:

```text
store base weights in NF4 / 4-bit
compute in BF16 if supported
train LoRA adapters in BF16/FP16
use paged or low-bit optimizers when memory-bound
```

## 16. Mental Model

The shortest way to remember mixed precision training is:

```text
Use low precision for big fast tensor ops.
Use higher precision where numerical stability matters.
Use BF16 when possible.
Use FP16 with loss scaling when necessary.
```

Mixed precision is usually the first training optimization to enable because it improves memory, bandwidth, and speed with relatively little code change.
