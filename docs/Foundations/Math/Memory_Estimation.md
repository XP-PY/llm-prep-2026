# How to Estimate GPU Memory for Model Training and Inference

## 1. First rule: parameter count times bytes per parameter

The most useful starting rule is:

$$
\text{memory} \approx \text{\#params} \times \text{bytes per param}
$$

The key is that **bytes per parameter depends on the workload**:

* only loading weights for inference
* full-parameter training
* LoRA / QLoRA fine-tuning

So the same 7B model may need **14 GB** in FP16 inference, but **100+ GB** for full fine-tuning.

---

## 2. Inference: only loading model weights

If the model is stored as:

* FP32: 4 bytes / param
* FP16 or BF16: 2 bytes / param
* INT8: 1 byte / param
* INT4: 0.5 byte / param

then the rough weight memory is:

$$
\text{weight memory} = P \times b
$$

where:

* $P$ = parameter count
* $b$ = bytes per parameter

### Example: a 7B model

* FP32: $7 \times 4 = 28$ GB
* FP16/BF16: $7 \times 2 = 14$ GB
* INT8: $7 \times 1 = 7$ GB
* INT4: $7 \times 0.5 = 3.5$ GB

### Fast inference rules

* **1B params in FP16/BF16 ≈ 2 GB**
* **1B params in FP32 ≈ 4 GB**
* **1B params in INT8 ≈ 1 GB**
* **1B params in INT4 ≈ 0.5 GB**

This only covers **weight storage**. Real deployment memory is usually larger because of **KV cache**, runtime buffers, and batch-size effects.

---

## 3. Full-parameter training / full fine-tuning

Training needs much more than weights. With mixed precision + Adam-style optimization, you usually store:

* model weights
* gradients
* FP32 master weights
* Adam states $(m, v)$
* activations

A common parameter-side estimate is:

$$
\text{training memory} \approx 16 \text{ bytes per param}
$$

Why 16?

$$
2 \text{ (weights)} + 2 \text{ (grads)} + 4 \text{ (FP32 master)} + 8 \text{ (Adam } m,v\text{)} = 16
$$

So:

$$
\text{memory} \approx 16P
$$

### Examples

* **0.6B**: $0.6 \times 16 = 9.6$ GB, plus overhead often becomes about **12 GB**
* **1.5B**: $1.5 \times 16 = 24$ GB
* **7B**: $7 \times 16 = 112$ GB

This is why a “full fine-tuning VRAM” table is much larger than a pure inference table.

---

## 4. LoRA training

For LoRA, the base model is still loaded, but only a small adapter is trainable.

So the rough memory becomes:

$$
\text{LoRA memory} \approx \text{base model weights} + \text{LoRA trainable states} + \text{activations}
$$

A useful approximation is:

$$
\text{LoRA memory} \approx 2P + 16P_{\text{LoRA}} + \text{activation overhead}
$$

where:

* $P$ = base model params
* $P_{\text{LoRA}}$ = LoRA params
* $2P$ = frozen backbone loaded in FP16/BF16
* $16P_{\text{LoRA}}$ = LoRA weights + grads + optimizer states

### Example: 7B backbone + 28M LoRA params

Base weights:

$$
7 \times 2 = 14 \text{ GB}
$$

LoRA trainable part:

$$
28\text{M} \times 16 \approx 448\text{ MB}
$$

So parameter-side memory is about:

$$
14.4 \text{ GB}
$$

Then add activations, temporary buffers, and fragmentation. In practice this may become **20-30 GB**.

The key point is:

> For LoRA, memory is **not** determined by LoRA params alone.  
> The frozen backbone still dominates.

---

## 5. Why real memory is bigger than the formula

The formulas above are only rough estimates because actual GPU memory also includes:

* **activations** during training
* **KV cache** during generation
* attention workspace / CUDA kernel buffers
* memory fragmentation
* sequence length and batch size effects

So parameter-based formulas are best viewed as:

* a **lower bound**, or
* a **quick rule of thumb**

not an exact runtime number.

---

## 6. How activations and KV cache scale

This is the part most people underestimate.

### 6.1 Activations in training

During training, activation memory grows roughly with:

$$
\text{activation memory} \propto B \times T \times L \times H \times \text{bytes}
$$

where:

* $B$ = batch size
* $T$ = sequence length
* $L$ = number of layers
* $H$ = hidden size

The exact constant depends on implementation details such as:

* whether you save activations for backward
* whether you use gradient checkpointing
* fused kernels / FlashAttention
* optimizer and framework internals

But the most important scaling rule is:

> If you double **batch size** or **sequence length**, activation memory usually grows almost linearly.

So even if parameter memory stays the same, training can suddenly OOM when:

* sequence length goes from 2K to 8K
* micro-batch goes from 1 to 4

Gradient checkpointing helps by recomputing activations instead of storing all of them, which trades **more compute** for **less memory**.

### 6.2 KV cache in autoregressive generation

For decoding, the main extra memory is the **KV cache**.

For a transformer layer, each token stores:

* one **key**
* one **value**

So a rough KV-cache formula is:

$$
\text{KV cache} \approx B \times T \times L \times 2 \times H_{kv} \times d_{head} \times \text{bytes}
$$

where:

* $B$ = batch size
* $T$ = current context length
* $L$ = number of layers
* $H_{kv}$ = number of KV heads
* $d_{head}$ = head dimension

Important consequence:

> KV cache grows roughly linearly with **batch size** and **context length**.

That is why long-context serving can run out of memory even when the model weights fit easily.

### 6.3 Why GQA / MQA helps deployment

In standard MHA:

* $H_{kv}$ is usually equal to the full number of attention heads

In GQA or MQA:

* multiple query heads share fewer KV heads
* so $H_{kv}$ becomes much smaller

That directly reduces KV-cache memory.

### 6.4 A quick KV-cache example

Suppose:

* $L = 32$ layers
* $H_{kv} = 32$
* $d_{head} = 128$
* BF16 cache: 2 bytes

Then per token, per sequence:

$$
32 \times 2 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} \approx 0.5 \text{ MB}
$$

So at 4K tokens, one sequence already needs about:

$$
0.5 \text{ MB} \times 4096 \approx 2 \text{ GB}
$$

If the model uses GQA with only $H_{kv}=8$, the KV cache drops to about **one quarter** of that.

### 6.5 The practical takeaway

* **Weight memory** depends mainly on model size and dtype.
* **Activation memory** depends strongly on training-time batch size and sequence length.
* **KV cache** depends strongly on serving-time batch size and context length.

So:

> training OOM is often an **activation problem**, while long-context inference OOM is often a **KV-cache problem**.

---

## 7. Useful mental model

If you only want a quick answer:

### Inference

* **1B params FP16/BF16 ≈ 2 GB**
* **1B params INT8 ≈ 1 GB**
* **1B params INT4 ≈ 0.5 GB**

### Full fine-tuning with Adam

* **1B params ≈ 16 GB**

### LoRA fine-tuning

* roughly **base model inference memory + a bit extra**
* much cheaper than full fine-tuning

---

## 8. Why many VRAM tables look the way they do

Many tables implicitly use:

* **full fine-tuning**: about **16 bytes/param**
* **LoRA fine-tuning**: frozen backbone + small trainable overhead + activations

That is why numbers like these are common:

* **7B full fine-tuning**: $7 \times 16 = 112$ GB
* **1.5B full fine-tuning**: $1.5 \times 16 = 24$ GB
* **0.6B full fine-tuning**: $0.6 \times 16 \approx 9.6$ GB, often rounded upward with overhead

So when reading a table, always ask:

1. Is this **inference** or **training**?
2. What **dtype** is used?
3. Does the number include **activations / KV cache / optimizer states**?

Without those details, VRAM numbers are easy to misread.
