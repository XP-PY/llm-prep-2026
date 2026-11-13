# Common data types

## Floating-point types

### FP32 (float32)

* **Format:** 1 sign, 8 exponent, 23 mantissa bits.
* **Pros:** Most stable numerically; good for reference/baselines and sensitive computations.
* **Cons:** Highest memory + bandwidth cost (4 bytes/param).

### FP16 (float16, IEEE half)

* **Format:** 1 sign, 5 exponent, 10 mantissa bits.
* **Pros:** 2× smaller than FP32 (2 bytes/param), faster on GPUs with tensor cores.
* **Cons:** Narrow exponent range → **overflow/underflow** more likely; training often needs **loss scaling** to avoid gradient underflow.

### BF16 (bfloat16)

* **Format:** 1 sign, **8 exponent**, 7 mantissa bits.
* **Pros:** Same exponent width as FP32 → much better **dynamic range** than FP16, usually more stable for training; typically no loss scaling needed.
* **Cons:** Less mantissa precision than FP16, but in deep learning it’s usually fine.

**Rule of thumb:**

* If your GPU supports BF16 well, **BF16 is often the easiest “works out of the box” training dtype**.
* FP16 can be great, but is more likely to need extra care (loss scaling, stability tweaks).

### TF32 (TensorFloat-32)

* **What it is:** NVIDIA Ampere+ mode where **matmuls run in TF32** by default (unless disabled), using FP32 range but reduced mantissa precision (roughly ~10-bit mantissa behavior for matmul).
* **Use:** Speeds up FP32 training/inference for matmul-heavy workloads with usually small accuracy impact.
* **Note:** TF32 is a **compute mode**, not a storage dtype in your model weights.

### FP8 (E4M3 / E5M2)

* **What it is:** 8-bit floating formats used for very fast training/inference on newer GPUs (Hopper+), typically with special recipes (scaling, amax tracking).
* **Pros:** Big speed/memory wins.
* **Cons:** Requires specific hardware/software support and careful numerics.

---

## Integer / quantized types

### INT8

* **8-bit integer quantization** of weights (and sometimes activations).
* Used in “8-bit loading” and “8-bit optimizers.”
* Good memory savings, often minimal quality loss with proper quantization.

### INT4 (generic 4-bit)

* **4-bit quantization** (huge memory savings).
* Harder to do well than INT8; quantization error can be more noticeable.

---

## NF4 (NormalFloat4) — the QLoRA star

NF4 is a **special 4-bit quantization codebook** designed for neural network weights that tend to look roughly **normally distributed**.

Instead of mapping values to 16 uniformly spaced levels (like naive linear INT4), NF4 uses **16 levels placed to better match a normal distribution**, which reduces quantization error for typical weight distributions.

In QLoRA-style training:

* The **base model weights** are stored in **NF4 4-bit** (frozen).
* Computation is done by **dequantizing on the fly** (often into BF16/FP16 for matmuls).
* You train **LoRA adapters** (and maybe a few small extras) in higher precision.

### “Double quantization”

You’ll also see `bnb_4bit_use_double_quant=True`: it means the **quantization constants/scales** (metadata) are themselves quantized to save additional memory.

---

## “Compute dtype” vs “storage dtype” (super important)

In quantized setups you often have:

* **Storage dtype:** how weights are stored in memory (e.g., NF4/INT4/INT8)
* **Compute dtype:** the dtype used during matmul/attention (e.g., FP16 or BF16)
* **Optimizer/accumulation dtype:** where gradients / optimizer states live (often FP32-ish, or 8-bit optimizer states)

Example in QLoRA:

* Base weights stored in **NF4 (4-bit)**
* Matmuls computed in **BF16** (`bnb_4bit_compute_dtype=torch.bfloat16`)
* LoRA weights stored/trained in BF16/FP16

---

## Common dtypes you’ll see in HF configs

* `torch_dtype=torch.float16` or `torch.bfloat16`: load model weights in that dtype (non-quantized).
* `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`: store base weights quantized.
* `optim="paged_adamw_8bit"`: optimizer states quantized to reduce VRAM.

---

## Practical guidance

* **Training (non-quantized):** prefer **BF16** if available; else FP16 + loss scaling.
* **QLoRA:** use **NF4 + BF16 compute** if your GPU supports BF16; otherwise NF4 + FP16 compute.
* **Inference:** often FP16/BF16 for speed; quantized inference (INT8/INT4/NF4) when memory-bound.

If you tell me your GPU model (or just “supports BF16?” + VRAM), I can recommend a clean default combo (storage/compute/optimizer) and explain the likely stability pitfalls for your setup.
