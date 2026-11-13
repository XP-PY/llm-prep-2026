# Tensor Parallelism (TP)

## 1. Why this note exists

When a dense Transformer becomes too large for one GPU, the first question is usually:

> Can we split one layer itself across multiple GPUs?

That is exactly what **Tensor Parallelism (TP)** does.

The clean mental model is:

> TP shards the **weights and intermediate hidden dimensions inside one layer** across multiple GPUs, so one layer is computed collaboratively by several ranks.

This is different from:

* **Data Parallelism**: each GPU holds the whole model, but different data
* **Pipeline Parallelism**: different GPUs hold different layers
* **Expert Parallelism**: different GPUs hold different MoE experts

TP is especially common in large dense LLM training, and it also appears inside MoE systems together with [EP](./EP.md).

---

## 2. Core Idea

Consider one linear layer:

$$
Y = XW + b
$$

where:

* $X \in \mathbb{R}^{B \times H_{\text{in}}}$
* $W \in \mathbb{R}^{H_{\text{in}} \times H_{\text{out}}}$
* $Y \in \mathbb{R}^{B \times H_{\text{out}}}$

If `W` is too large for one GPU, TP shards `W` across `p` GPUs.

At a high level, there are two canonical shardings:

1. **Column Parallel**
2. **Row Parallel**

These two are the building blocks behind most TP implementations such as Megatron-LM style Transformer parallelism.

---

## 3. Column Parallel Linear

### 3.1 Formula

Split `W` along its **output dimension**:

$$
W = [W_1, W_2, \dots, W_p]
$$

where:

* $W_i \in \mathbb{R}^{H_{\text{in}} \times H_{\text{out}}/p}$

Then each rank computes:

$$
Y_i = XW_i
$$

and the full output is:

$$
Y = [Y_1, Y_2, \dots, Y_p]
$$

So each rank produces only a **slice of the output features**.

### 3.2 Intuition

Every rank sees the same input `X`, but only owns part of the columns of `W`.

That means:

* input is replicated
* output is sharded

### 3.3 Diagram

```text
Shared X
   |
   +--> W1 -> Y1
   +--> W2 -> Y2
   +--> ...
   +--> Wp -> Yp

Full Y = concat(Y1, Y2, ..., Yp)
```

### 3.4 Communication

In the simplest view, column parallel needs:

* **no communication before local matmul**
* **concatenation / all-gather after matmul** if the next operation needs the full `Y`

But in real Transformer implementations, the full `Y` is often **not gathered immediately**.

That is the important optimization.

For example, in an MLP:

$$
\text{MLP}(x) = W_2 \phi(W_1 x)
$$

If `W1` is column-parallel, then each rank can keep its own local slice:

$$
z_i = \phi(XW_{1,i})
$$

and feed it directly into a **row-parallel** `W2`, avoiding an unnecessary all-gather in the middle.

---

## 4. Row Parallel Linear

### 4.1 Formula

Split `W` along its **input dimension**:

$$
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_p
\end{bmatrix}
$$

where:

* $W_i \in \mathbb{R}^{H_{\text{in}}/p \times H_{\text{out}}}$

Then split the input consistently:

$$
X = [X_1, X_2, \dots, X_p]
$$

where:

* $X_i \in \mathbb{R}^{B \times H_{\text{in}}/p}$

Each rank computes:

$$
\tilde{Y}_i = X_i W_i
$$

and the full output is:

$$
Y = \sum_{i=1}^{p} \tilde{Y}_i
$$

So each rank produces a **partial contribution to the same output features**.

### 4.2 Intuition

Here the input is sharded and the output must be **reduced** across ranks.

That means:

* input is sharded
* output is logically shared

### 4.3 Diagram

```text
X1 -> W1 -> Y~1
X2 -> W2 -> Y~2
...
Xp -> Wp -> Y~p

Full Y = Y~1 + Y~2 + ... + Y~p
```

### 4.4 Communication

Row parallel typically needs:

* a consistent input sharding
* an **all-reduce** or **reduce-scatter** on the output

This is why row parallel often appears right after a column-parallel layer:
the previous layer already produced an output sharded in exactly the needed hidden dimension.

---

## 5. Why Column + Row Parallel Pair So Well

The standard dense Transformer MLP uses:

1. column-parallel for the first projection
2. row-parallel for the second projection

That gives:

$$
X
\xrightarrow{\text{column parallel } W_1}
Z
\xrightarrow{\phi}
\phi(Z)
\xrightarrow{\text{row parallel } W_2}
Y
$$

The nice property is:

* after column parallel, hidden states are already split across the expanded dimension
* row parallel naturally consumes that split
* only one synchronization is needed at the end

This is one of the main reasons TP is practical for Transformer blocks.

---

## 6. TP in Attention

TP is not only for MLPs. It is also very natural for multi-head attention.

Suppose:

* hidden size = `H`
* number of heads = `n_h`
* TP size = `p`

A common choice is to shard the attention heads:

$$
n_h \rightarrow \frac{n_h}{p} \text{ heads per rank}
$$

Each rank computes its own local:

* `Q`
* `K`
* `V`
* attention scores
* attended head outputs

because attention heads are mostly independent before the final output projection.

### 6.1 Local Attention Computation

For local heads on rank `i`:

$$
Q_i = XW_i^Q,\quad K_i = XW_i^K,\quad V_i = XW_i^V
$$

Then:

$$
\text{Attn}_i(X) = \operatorname{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right)V_i
$$

These local head outputs are concatenated logically across ranks and then passed through the final output projection.

### 6.2 Why Attention Fits TP Well

Because multi-head attention already factorizes computation by heads, splitting heads across GPUs is structurally natural.

The usual pattern is:

* QKV projections: column parallel
* output projection: row parallel

This mirrors the MLP pattern.

---

## 7. Communication Patterns in TP

The key engineering question in TP is not the math alone, but **where communication happens**.

### 7.1 Common Collectives

TP frequently uses:

* **All-gather**
  * collect sharded hidden states and concatenate them
* **All-reduce**
  * sum partial outputs or gradients
* **Reduce-scatter**
  * sum partial outputs while leaving the result sharded

### 7.2 Rule of Thumb

* **column parallel** tends to end with **gather-like** semantics
* **row parallel** tends to end with **sum-like** semantics

### 7.3 Why Communication Matters

TP reduces memory per GPU, but it introduces fine-grained communication on almost every layer.

So TP usually works best:

* within one node
* over fast interconnects such as NVLink / NVSwitch

If TP spans slow links, communication can dominate runtime.

---

## 8. Backward Pass Intuition

The same sharding logic also affects backward.

### Column Parallel

If:

$$
Y_i = XW_i
$$

then each rank can compute its own:

$$
\frac{\partial \mathcal{L}}{\partial W_i}
$$

locally from `X` and the local output gradient slice.

The subtle point is the input gradient:

$$
\frac{\partial \mathcal{L}}{\partial X}
=
\sum_{i=1}^{p}
\frac{\partial \mathcal{L}}{\partial Y_i} W_i^\top
$$

So the full input gradient requires contributions from all ranks.

### Row Parallel

If:

$$
Y = \sum_{i=1}^{p} X_i W_i
$$

then each rank computes local parameter gradients, and the output gradient is already shared conceptually. The exact communication pattern depends on whether hidden states are stored replicated or sharded between neighboring layers.

The big picture is:

> TP saves memory in forward, but the distributed graph must still reconstruct the right shared gradients in backward.

---

## 9. Memory and Compute Scaling

If a weight matrix `W` is sharded across `p` ranks, then the **parameter memory per rank** is roughly:

$$
\frac{1}{p}
$$

of the unsharded weight memory.

The local GEMM size is also reduced accordingly.

So TP helps with:

* parameter memory
* optimizer state memory
* local matmul size

But it does **not** make communication free. In practice:

* larger `p` gives more memory savings
* larger `p` also increases synchronization overhead

So TP size is usually chosen as a compromise.

---

## 10. Where TP Is Commonly Used

TP is especially common in:

* large dense Transformer training
* the dense parts of MoE models
* attention and MLP layers in Megatron-style training stacks

It is often combined with:

* **DP** for batch scaling
* [PP](./PP.md) for layer-wise sharding
* [EP](./EP.md) for MoE expert sharding

A common large-scale training mesh looks like:

```text
global world size
= DP x PP x TP
```

or for MoE:

```text
global world size
= DP x PP x TP x EP
```

---

## 11. Benefits

### Fits larger dense layers

Without TP, a very large projection matrix may not fit in one GPU's memory.

### Preserves exact dense computation

TP is not an approximation. It is just a distributed implementation of the same linear algebra.

### Works naturally with Transformer structure

Attention heads and MLP hidden expansions already have shard-friendly structure.

---

## 12. Trade-offs

### Frequent communication

TP introduces collectives inside many layers, not just between stages.

### Best on fast interconnects

It is usually much more efficient inside one machine than across a slow network.

### Smaller local matmuls can hurt efficiency

If TP size becomes too large, each rank's GEMM becomes too small to fully utilize the hardware.

So "more TP" is not always better.

---

## 13. TP vs Other Parallelism

| Aspect | Data Parallel | Tensor Parallel | Pipeline Parallel | Expert Parallel |
|:---|:---|:---|:---|:---|
| What is split? | batch | tensors inside a layer | layers / stage blocks | experts in MoE |
| Each rank holds full dense layer? | yes | no | only some layers | only some experts |
| Main communication | gradient all-reduce | hidden-state collectives | stage-to-stage send/recv | token all-to-all |
| Best for | more data throughput | huge dense layers | huge model depth | huge MoE expert count |

---

## 14. Key Takeaways

1. **TP shards the weights and hidden dimensions inside one layer across GPUs**
2. **Column parallel shards output features; row parallel shards input features**
3. **The standard Transformer pattern is column-parallel first, row-parallel second**
4. **TP saves memory and enables larger dense models, but it introduces frequent communication**
5. **TP is most effective when used over very fast interconnects and combined with DP / PP / EP at larger scale**
