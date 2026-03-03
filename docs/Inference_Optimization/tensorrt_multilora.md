# TensorRT-LLM & Multi-LoRA Serving

## 1) What problem are we solving?

When you deploy LLMs, you usually want **two things at the same time**:

1. **Maximum GPU throughput / lowest latency**
2. **Personalization** (different users/teams/products need different fine-tunes)

These two goals often conflict:

* “Personalization” usually means **many model variants** → expensive to load/serve.
* “Max throughput” usually means **one highly-optimized engine** → hard to modify per request.

**TensorRT-LLM** targets (1).
**Multi-LoRA serving** targets (2).
Together, they aim for: **one compiled engine + thousands of adapters**.

---

## 2) TensorRT-LLM in one sentence

**TensorRT-LLM** is NVIDIA’s inference stack that **compiles** Transformer models into highly optimized GPU execution graphs (TensorRT engines) using:

* kernel fusion,
* specialized attention/KV-cache implementations,
* low-precision math (FP8/INT4),
* custom plugins for Transformer ops.

Think:

* **HF Transformers**: flexible, easy, but more Python overhead and less aggressive fusion.
* **vLLM**: extremely strong throughput in Python ([continuous batching + paged KV cache](./continuous_batching.md)).
* **TensorRT-LLM**: “compile and squeeze the GPU” for peak performance—especially on NVIDIA datacenter GPUs.

> Important: “2–5x faster than vLLM” is **not guaranteed**. It depends on: batch shape, context length, model size, GPU (A100 vs H100), precision (FP16/BF16/FP8/INT4), and whether you’re throughput-bound or latency-bound.

---

## 3) Why TensorRT-LLM can be faster (theory intuition)

### 3.1 Kernel fusion → fewer launches, less memory traffic

Transformers are full of small ops. In naive execution you get:

* many kernel launches,
* intermediate tensors written to HBM,
* extra synchronization.

TensorRT tries to:

* fuse ops,
* reuse memory,
* schedule kernels to better utilize SMs and Tensor Cores.

### 3.2 Custom plugins for Transformer “hot paths”

Some operations are performance-critical and not ideal in standard graph lowering:

* **Attention** (especially long-context)
* **KV cache** updates
* **RoPE** (rotary embeddings)
* **GQA/MQA** variants
* **Fused MLP** (GELU/SwiGLU + linear layers)

TensorRT-LLM often uses dedicated plugins that are closer to hand-tuned CUDA.

### 3.3 Low precision on modern GPUs

On H100/A100, using FP8 or INT4 can shift you from being **memory-bound** to **compute-efficient** (or at least reduce bandwidth pressure). The effective gain depends on:

* weight-only quant vs activation quant,
* KV cache precision,
* accuracy constraints.

---

## 4) Multi-LoRA serving: what it is and why it matters

### 4.1 The idea

You keep a **single base model** (compiled/loaded once), and for each request you choose an **adapter** (LoRA) that applies a small delta to some layers.

This means:

* you don’t load 1,000 full checkpoints,
* you load 1 base engine + many tiny LoRA weights,
* requests can “route” to different behaviors.

### 4.2 The LoRA math (single adapter)

For a Linear layer with base weight (W):
$$
y = xW^\top
$$
LoRA adds a low-rank update:
$$
W' = W + \Delta W,\quad \Delta W = s\cdot BA
$$
where:

* $A \in \mathbb{R}^{r \times d_{in}},; B \in \mathbb{R}^{d_{out} \times r}$
* $r$ is small (e.g., 4/8/16)
* $s = \alpha / r$ (typical scaling)

So the forward becomes:
$$
y = xW^\top + s\cdot xA^\top B^\top
$$

### 4.3 Multi-LoRA per request (routing)

There are two common “multi-adapter” modes:

**Mode A: one adapter per request** (most common in production)
$$
y = \text{Base}(x) + s_{k}\cdot \text{LoRA}_{k}(x)
$$
where request selects adapter $k$.

**Mode B: mixture of adapters** (less common; can be used for composition)
$$
y = \text{Base}(x) + \sum_{i=1}^{m} \alpha_i \cdot \text{LoRA}_i(x)
$$
This is useful for “skill composition”, but is harder to optimize.

### 4.4 Why this is powerful in real systems

* **Personalized assistants**: same model, different org policies/terminology.
* **Multi-tenant serving**: one GPU fleet supports many fine-tunes.
* **A/B testing**: route traffic to adapters, compare metrics, roll back instantly.

---

## 5) Real-world serving picture (end-to-end)

A typical production stack looks like:

1. **Build/compile** a TensorRT engine for the **base model** (once per target GPU + precision + max shapes).
2. Store a **LoRA adapter library** (many small adapter files).
3. Online serving:

   * each request carries `adapter_id` (or user/org mapping),
   * server loads/caches adapter weights (often on GPU),
   * scheduler batches requests (ideally grouping by adapter or using efficient switching),
   * run inference with base engine + adapter deltas.

---

## 6) Pitfalls (practical “gotchas” you should expect)

### 6.1 Engine build and shape constraints

* Building engines can be **slow** (minutes to hours) depending on:

  * model size (e.g., 70B),
  * optimization profiles,
  * precision calibration,
  * target GPU.
* TensorRT engines often require **predefined ranges**:

  * max batch size,
  * max sequence length,
  * max KV cache sizes.
    If your production traffic exceeds the profile, performance can degrade or fail.

### 6.2 Adapter switching overhead

If requests randomly alternate between adapters:

* you may lose batching efficiency,
* you may thrash GPU memory/caches if adapters aren’t cached well,
* you may pay extra kernel/setup overhead.

**Mitigation**:

* batch/group by adapter (when possible),
* keep “hot adapters” resident on GPU,
* cap the number of simultaneously active adapters per GPU.

### 6.3 Quantization + LoRA interactions

* With INT4/FP8, the base weight is quantized.
* LoRA deltas are usually in higher precision (FP16/BF16).
  This mixed-precision path can be tricky:
* accuracy regressions,
* performance regressions if LoRA application is not fused.

### 6.4 Debuggability & iteration speed

* HF/vLLM are easier to debug quickly.
* TensorRT-LLM is more “production engineering”: compilation, profiles, plugins.
  A common workflow is:
* prototype on HF/vLLM,
* validate correctness + prompts + adapters,
* then “graduate” to TensorRT-LLM for production performance.

---

## 7) Comparison table (more “engineering-realistic”)

| Stack                  | What it optimizes     | Typical strengths                                                      | Typical weaknesses                                                                 | Multi-LoRA maturity                               |
| ---------------------- | --------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------- |
| Hugging Face (PyTorch) | flexibility           | easiest debugging, fastest iteration                                   | slower throughput, Python overhead                                                 | via PEFT; not a serving solution                  |
| vLLM                   | throughput + batching | excellent throughput, paged KV cache, fast to deploy                   | not always peak single-request latency; GPU-specific optimizations less “compiled” | mature (LoRA routing supported)                   |
| TensorRT-LLM           | peak GPU efficiency   | strong FP8/INT4, plugin-optimized attention/MLP, low latency potential | engine build complexity, shape/profile constraints                                 | emerging / improving (depends on version/plugins) |

> Takeaway: **vLLM is the best default serving baseline**. Use **TensorRT-LLM** when you are sure performance gains justify compilation + ops complexity.

---

## 8) Engine building: what actually happens (detailed but readable)

A simplified “mental pipeline”:

1. **Model ingestion**

   * Start with an HF checkpoint (weights + config).
2. **Graph lowering**

   * Convert the model computation into a representation TensorRT can optimize.
3. **Plugin insertion**

   * Replace key subgraphs with custom kernels/plugins:

     * attention,
     * RoPE,
     * KV-cache management,
     * fused MLP, etc.
4. **Optimization profiles**

   * Decide ranges for:

     * batch size,
     * input length,
     * output length,
     * cache size.
5. **Quantization (optional)**

   * FP8/INT4 paths (weight-only or more aggressive quant).
6. **Build engine**

   * TensorRT compiles kernels, fuses, plans memory, and emits an engine plan.
7. **Runtime**

   * Serving uses the engine plan + runtime scheduler to execute inference.

---

## 9) Multi-LoRA implementation patterns (what “dynamic adapters” really mean)

### Pattern 1: “Adapter as extra weights” (fastest if supported)

At runtime, LoRA matrices (A,B) are provided to kernels (or stored in GPU memory) and applied inside a fused path.

### Pattern 2: “Pre-merge LoRA into weights” (fast but not dynamic)

Compute:
$$
W_{\text{merged}} = W + s\cdot BA
$$
This is great for one adapter, but you lose the “multi-tenant” benefit unless you rebuild or store many merged engines.

### Pattern 3: “Hybrid caching”

Keep base engine fixed; cache a limited number of merged weights for “hot adapters”.
Good when adapter set is large but traffic is concentrated.

---

## 10) Step-by-step demo outline (repo-friendly)

#### Cell 1: Install

```bash
pip install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com
```

#### Cell 2: Build engine (conceptual example)

```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8b",
    quantization="fp8",   # or "int4" depending on hardware/support
)
llm.save_engine("llama3-8b-trt")
```

#### Cell 3: Multi-LoRA routing (pseudo-API)

```python
# PSEUDO-CODE: APIs differ by version / examples.
llm = LLM(
    model="base_engine_path",
    lora_paths=["lora_legal", "lora_med", "lora_customerA"]
)

prompts = ["...", "...", "..."]
adapter_ids = [0, 2, 2]  # per request adapter routing

outputs = llm.generate(
    prompts,
    lora_indices=adapter_ids
)
```

#### Cell 4: Benchmark methodology (don’t just “time one run”)

Measure at least:

* **prefill** latency (input processing)
* **decode** throughput (tokens/sec)
* **end-to-end** latency (P50/P95)
* GPU utilization + VRAM peak

Minimal benchmark skeleton:

```python
import time
import torch

def benchmark(generate_fn, prompts, warmup=3, iters=10):
    for _ in range(warmup):
        _ = generate_fn(prompts)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = generate_fn(prompts)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters
```