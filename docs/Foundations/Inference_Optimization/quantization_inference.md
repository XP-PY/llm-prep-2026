# Post-Training Quantization (GPTQ / AWQ)

## 1) Why quantization matters

Large LLMs are usually stored and served in **FP16/BF16** (16-bit).
That keeps quality high, but **weights dominate memory**.

Example intuition:

* A 70B-parameter model in FP16 needs roughly:
  $$
  70\text{B params} \times 2\text{ bytes} \approx 140\text{ GB}
  $$

So a single GPU cannot hold it unless you do sharding/offload.

**Post-Training Quantization (PTQ)** compresses weights after training:

* **4-bit / 3-bit / 2-bit weights**
* Typical benefit: **~4× memory reduction at 4-bit** (often the sweet spot)
* Usually a **small quality drop** if done carefully

---

## 2) What PTQ actually does (mental model)

Weights are real numbers (float). Quantization replaces them with low-bit integers:

1. Choose a **scale** (S) (and sometimes zero-point)
2. Convert float weights (W) to integers:
   $$
   Q = \text{round}(W / S)
   $$
3. During inference, compute using int weights (or int-dequant fused kernels):
   $$
   \hat{W} = Q \cdot S
   $$

If you quantize naively, you lose accuracy. GPTQ and AWQ are “smart PTQ” methods that decide **how to quantize** with minimal damage.

---

## 3) Two key PTQ methods

### A) [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323) — “minimize the damage per layer”

**Idea:** quantize one layer at a time, but choose quantized weights so that the layer’s output changes as little as possible.

* It uses a second-order approximation (Hessian / curvature) to estimate which weight errors matter most.
* It quantizes columns (or groups) sequentially and **compensates** the remaining weights to reduce accumulated error.

**Why it works:**
Not all weight errors are equally harmful. GPTQ tries to put quantization error where it hurts least.

---

### B) [AWQ (Lin et al., 2023)](https://arxiv.org/abs/2306.00978) — “protect important weights using activations”

**Observation:** activations often have **outliers** (very large values).
When large activations pass through certain weight channels, small weight errors get amplified.

**Idea:** use activation statistics from a small calibration set to:

* **scale channels** to make quantization friendlier
* **protect “salient” weights** (often top ~1% per channel/group) from being overly distorted

**Why it works:**
It directly optimizes for keeping (XW) (the layer output on real activations (X)) close after quantization.

---

## 4) What “calibration” means (Quantization)

**Calibration** is a short “measurement” phase done **after training** but **before quantizing**.
You run the original FP16/BF16 model on a **small, representative sample of inputs** to learn how the model’s values behave, so you can pick good quantization settings.

### What you do during calibration

1. **Prepare calibration samples**

   * Usually a few hundred to a few thousand text sequences
   * Ideally from the same domain as your real usage (chat, code, medical text, etc.)
   * These samples are **not used to train** the model (no backprop)

2. **Run forward passes (inference only)**

   * Feed those texts through the FP16 model
   * No gradients, no weight updates

3. **Collect statistics (“what ranges do we see?”)**
   Typical stats include:

   * **Activation ranges / outliers** per layer or per channel
     (e.g., max absolute value, percentile ranges, how extreme the outliers are)
   * **Input distributions** to each linear layer (often used for better scaling/clipping)
   * For **GPTQ-like methods**, information that approximates “sensitivity” of weights
     (often described as Hessian/covariance-related signals)

4. **Choose quantization parameters**
   Using those statistics, the quantizer decides:

   * **Scales** (how floats map to low-bit integers)
   * **Clipping thresholds** (how aggressively to clip outliers before quantizing)
   * **Grouping strategy** (e.g., per-tensor vs per-channel, group size like 64/128)
   * (AWQ) **Which channels/weights are “salient”** and should be protected via scaling

### Why calibration matters

Quantization error depends heavily on **value ranges and outliers**.
If you pick scales/clipping blindly:

* you may waste precision on rare extremes, or
* clip important values and damage accuracy

Calibration helps the quantizer make these choices using **real model behavior** on real-ish inputs, so quality drop is much smaller.

### Cost compared to training

Calibration is cheap because it is:

* **forward pass only**
* on a **small dataset**
* typically finishes in **minutes** (not days)

**One-line summary:** calibration is “profiling the model’s activations on typical inputs” so your 4/3/2-bit quantization picks the right scales and doesn’t break the model.


---

## 5) Real-world results (rule of thumb)

* **4-bit** quantization is the most common “good trade-off”
* Often enables:

  * **7B–13B on consumer GPUs easily**
  * **70B on ~24–48GB GPUs** with efficient kernels + careful config (workload dependent)

3-bit and 2-bit can work, but become much more fragile.

---

## 6) Comparison table (clean)

| Method    | Bit width | Memory / speed vs FP16 |             Quality drop (typical) | Per-channel scaling | Salient weight protection | Common tooling        |
| --------- | --------: | ---------------------: | ---------------------------------: | ------------------: | ------------------------: | --------------------- |
| FP16/BF16 |        16 |                     1× |                               None |                  No |                       N/A | Baseline              |
| GPTQ      |     4 / 3 |                  ~3–4× |         Low (often small PPL rise) |                 Yes |             Hessian-aware | AutoGPTQ, ExLlama     |
| AWQ       |     4 / 3 |                  ~3–4× | Very low (often smaller than GPTQ) |                 Yes |          Activation-aware | AutoAWQ, vLLM support |

*(Exact numbers depend heavily on model, dataset, kernel implementation, and quant config.)*

---

## 7) “Derivations” in plain language

### GPTQ: layer-wise “optimal” rounding

For one layer weight matrix (W), GPTQ aims to choose a quantized (Q) such that the error matters little **under the layer’s sensitivity**:
$$
\min_Q |W - Q|_H^2
$$

* $H$ represents “importance / curvature”: errors along sensitive directions get penalized more.

Practical behavior:

* quantize some columns first
* update/compensate the remaining columns to reduce future error accumulation

You can think of it as: **round carefully, and correct the leftovers.**

---

### AWQ: keep real activations’ outputs stable

AWQ focuses on preserving the layer outputs for typical inputs $X$:
$$
S = \arg\min |XW - XQ(W)|
$$
It chooses scaling (S) (often per channel / per group) so that quantization doesn’t distort the important activation pathways.

You can think of it as: **quantize in the coordinate system that the activations actually use.**

---

## 8) Pitfalls (where PTQ breaks)

* **INT3 / INT2 instability** without careful clipping/scaling
* Some components are sensitive:

  * embeddings
  * layer norms
  * small projection layers in attention/MLP depending on architecture
* Bad group size / clip settings can cause sudden quality collapse (“it seems fine then breaks”).

---

## 10) Minimal “how you’d use it” (conceptual code)

### GPTQ-style (concept)

* load FP16 model
* run calibration samples
* quantize weights layer-by-layer
* save quantized checkpoint

### AWQ-style (concept)

* load FP16 model
* collect activation stats on calibration samples
* search per-group scaling and clip
* quantize weights
* save quantized checkpoint