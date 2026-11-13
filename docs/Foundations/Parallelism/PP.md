# Pipeline Parallelism (PP)

## 1. Why this note exists

If **Tensor Parallelism** splits one layer across GPUs, then the next natural idea is:

> Can we split the **stack of layers** itself across GPUs?

That is exactly what **Pipeline Parallelism (PP)** does.

The clean mental model is:

> PP divides the model into several consecutive layer groups called **stages**, and each stage lives on a different GPU or GPU group.

So if a Transformer is too deep or too large to fit on one device, PP lets different devices own different parts of the network.

This is different from:

* **TP**: split within one layer
* **DP**: replicate the whole model and split data
* **EP**: split experts in MoE

---

## 2. Core Idea

Suppose the model is:

$$
f(x) = f_S \circ f_{S-1} \circ \cdots \circ f_2 \circ f_1(x)
$$

where the full network is partitioned into `S` pipeline stages.

Then:

* stage 1 holds `f_1`
* stage 2 holds `f_2`
* ...
* stage `S` holds `f_S`

During forward:

$$
x
\xrightarrow{f_1}
h_1
\xrightarrow{f_2}
h_2
\xrightarrow{\cdots}
\xrightarrow{f_S}
y
$$

The intermediate activations are sent from one stage to the next.

So PP primarily reduces:

* model memory per device

but it introduces:

* inter-stage communication
* pipeline scheduling complexity

---

## 3. The Simplest Picture

Imagine a 24-layer Transformer and 4 GPUs.

One simple partition is:

```text
GPU0: layers  1 -  6
GPU1: layers  7 - 12
GPU2: layers 13 - 18
GPU3: layers 19 - 24
```

If one batch is sent through this naively:

1. GPU0 runs first
2. GPU0 sends activations to GPU1
3. GPU1 runs and sends to GPU2
4. GPU2 runs and sends to GPU3
5. GPU3 produces output

This already works, but it wastes hardware because only one stage is busy at a time.

That is why PP almost always uses **micro-batching**.

---

## 4. Micro-Batches and the Pipeline

Instead of sending one large batch through the whole model at once, we split a batch into `m` **micro-batches**:

$$
\mathcal{B} = \{\mathcal{B}_1, \mathcal{B}_2, \dots, \mathcal{B}_m\}
$$

Then the stages can work on different micro-batches simultaneously.

Example with 4 stages:

```text
time 1: stage1 -> micro-batch1
time 2: stage1 -> micro-batch2, stage2 -> micro-batch1
time 3: stage1 -> micro-batch3, stage2 -> micro-batch2, stage3 -> micro-batch1
time 4: stage1 -> micro-batch4, stage2 -> micro-batch3, stage3 -> micro-batch2, stage4 -> micro-batch1
...
```

This is exactly why it is called a pipeline: different stages process different micro-batches concurrently.

---

## 5. Pipeline Bubble

PP improves utilization, but it introduces **bubble time**:

* at the beginning, later stages are idle because the first micro-batch has not reached them yet
* at the end, earlier stages become idle while the last micro-batches finish downstream

This idle region is called the **pipeline bubble**.

A useful rule-of-thumb formula is:

$$
\text{bubble fraction}
\approx
\frac{S - 1}{m + S - 1}
$$

where:

* `S` = number of pipeline stages
* `m` = number of micro-batches

Interpretation:

* larger `m` reduces the bubble
* larger `S` increases the bubble

So deeper pipelines generally need more micro-batches to stay efficient.

---

## 6. Forward and Backward Under PP

Training is not only forward. We also need backward.

For one micro-batch `j`, the usual training objective is:

$$
\mathcal{L}^{(j)} = \mathcal{L}(f(x^{(j)}), y^{(j)})
$$

and the full mini-batch loss is often accumulated across micro-batches:

$$
\mathcal{L}
=
\frac{1}{m}
\sum_{j=1}^{m}
\mathcal{L}^{(j)}
$$

During backward:

* later stages compute gradients first
* gradient activations are sent backward to earlier stages

So PP needs communication in **both directions**:

* forward activations move left to right
* backward gradients move right to left

---

## 7. Main Pipeline Schedules

The difference between practical PP systems is often the **schedule**.

### 7.1 Naive Sequential Schedule

Do full forward of the whole batch, then full backward.

This is easy to understand, but utilization is poor.

### 7.2 GPipe Schedule

GPipe popularized the idea:

1. split the batch into micro-batches
2. run all forward micro-batches through the pipeline
3. then run all backward micro-batches

This gives much better utilization than naive sequential execution.

But GPipe must store many forward activations until backward starts, so activation memory can be large.

### 7.3 1F1B Schedule

`1F1B` means:

* one forward
* one backward

After warm-up, each stage alternates between forward and backward work on different micro-batches.

This reduces activation memory relative to GPipe because backward starts earlier.

The intuition is:

* GPipe prioritizes simpler scheduling
* 1F1B prioritizes lower memory and better overlap

### 7.4 Interleaved Pipeline

Some systems split each physical stage into multiple smaller virtual chunks.

This is called **interleaved PP**.

It helps because:

* stage granularity becomes finer
* load balance can improve
* bubble can shrink

But it also makes scheduling more complex.

---

## 8. Stage Partitioning

PP sounds simple, but one practical question is surprisingly important:

> Which layers should go to which stage?

The most obvious split is equal number of layers per stage, but that is not always best.

Why?

Because different layers may have different cost:

* embedding layers can be unusually large
* attention and MLP costs can differ
* MoE layers can be much more expensive than dense layers
* first/last stages may include extra work such as token embedding or logits

If one stage is much slower than the others, the whole pipeline slows down.

So stage partitioning should ideally balance:

* parameter memory
* activation memory
* forward compute
* backward compute
* communication volume

---

## 9. Communication in PP

PP communication is simpler than TP or EP conceptually.

It mainly involves:

* **send / recv activations** between neighboring stages in forward
* **send / recv gradients** between neighboring stages in backward

So unlike TP:

* PP does not usually need all-reduce inside every layer

and unlike EP:

* PP does not do global all-to-all token routing

This is why PP can scale across machines more naturally than very fine-grained TP in many settings.

Still, activations can be large, especially for:

* long sequence length
* large hidden size
* many micro-batches in flight

So PP is not communication-free. It just communicates at **stage boundaries** instead of within every layer.

---

## 10. Memory Trade-offs

PP helps because each stage stores only a subset of layers.

So parameter memory per rank is roughly reduced from:

$$
\text{full model memory}
\quad \to \quad
\text{about } \frac{1}{S}
$$

if stages are balanced.

But PP introduces other memory costs:

* stored activations for in-flight micro-batches
* optimizer state for local layers
* possible buffer memory for send / recv

This is why PP is often paired with:

* activation checkpointing
* 1F1B scheduling
* careful micro-batch tuning

---

## 11. Throughput vs Latency

PP is mainly a **throughput-oriented** strategy for large training runs.

Why?

Because:

* the whole model no longer fits on one device
* many micro-batches can keep stages busy

But PP also increases end-to-end latency for one sample because the sample must traverse multiple stages.

So PP is usually great when:

* training very large models
* maximizing overall tokens/sec

and less attractive when:

* single-request latency is the primary concern

---

## 12. PP in Inference

Pipeline ideas also appear in inference, but the trade-offs differ.

For autoregressive decoding:

* each token must still pass through all stages
* communication happens every decoding step

So PP inference can help fit a huge model, but it does not automatically make decoding fast.

In practice:

* PP is often essential for fitting very large checkpoints
* but latency-sensitive inference often prefers minimizing pipeline depth when possible

---

## 13. PP vs TP

This comparison is important because TP and PP are often used together.

### Tensor Parallelism

* splits **within a layer**
* fine-grained communication inside many layers
* best over very fast links

### Pipeline Parallelism

* splits **across layers**
* communication only between neighboring stages
* naturally useful when model depth is the dominant memory problem

You can remember it like this:

* **TP**: many GPUs compute one layer together
* **PP**: different GPUs compute different layers in sequence

---

## 14. PP vs Data Parallelism

With DP:

* every rank holds the whole model
* different ranks process different data

With PP:

* no rank holds the whole model
* one sample flows through multiple ranks

So PP is usually introduced when DP alone is impossible because the model is too large to replicate fully.

---

## 15. PP in Large Training Systems

At large scale, PP is rarely used alone.

A common setup is:

```text
world size = DP x PP x TP
```

or for MoE:

```text
world size = DP x PP x TP x EP
```

This means:

* PP handles model depth
* TP handles large dense layers
* EP handles many experts
* DP handles overall throughput

Modern large-model training is often about choosing the right combination, not one single parallelism type.

---

## 16. Benefits

### Fits deeper / larger models

PP is often the simplest way to split a model that is too large to place on one device.

### Communication is stage-local

Only neighboring stages exchange activations and gradients.

### Works naturally with model depth

Transformers are already stacked layer-by-layer, so stage partitioning is conceptually straightforward.

---

## 17. Trade-offs

### Pipeline bubble

Idle time appears during warm-up and drain.

### Scheduling complexity

Good PP needs micro-batching and a careful execution schedule.

### Load imbalance

The slowest stage can bottleneck the whole system.

### Activation memory

Depending on schedule, many micro-batch activations may need to be kept alive.

---

## 18. Key Takeaways

1. **PP splits a model by layers into several sequential stages**
2. **Micro-batching is essential; otherwise hardware utilization is poor**
3. **The main inefficiency is the pipeline bubble, which shrinks as the number of micro-batches grows**
4. **GPipe and 1F1B are the two most important scheduling ideas to know**
5. **PP is usually combined with TP, DP, and sometimes EP in modern large-model training**
