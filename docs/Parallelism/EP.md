# Expert Parallelism (EP)

## 1. Why this note exists

For dense models, the main memory problem is usually:

* too many dense parameters per layer

For MoE models, the problem changes:

* there may be **many experts**, but only a few are used for each token

That leads to a new question:

> If only a small subset of experts is used per token, can we shard the experts themselves across GPUs?

That is exactly what **Expert Parallelism (EP)** does.

The clean mental model is:

> EP distributes different MoE experts across different GPUs, then routes tokens to the GPUs that own the selected experts.

This is one of the key techniques behind large open MoE systems such as Mixtral and [DeepSeek-V2](../Large_Models/DeepSeek_V2.md).

---

## 2. MoE Reminder: What Needs to Be Parallelized?

A standard sparse MoE layer can be written schematically as:

$$
y
=
\sum_{e \in \mathcal{T}(x)}
g_e(x)\,E_e(x)
$$

where:

* $E_e(\cdot)$ is expert `e`
* $g_e(x)$ is the router weight for expert `e`
* $\mathcal{T}(x)$ is the top-`k` selected experts for token `x`

For example:

* `top-1` in Switch Transformer
* `top-2` in many later MoE models

The crucial point is:

* the total number of experts can be very large
* but each token only activates a small subset

So storing all experts on every GPU is wasteful. EP avoids that replication.

---

## 3. Core Idea

Suppose a MoE layer has:

* `E` routed experts
* `P` GPUs in the EP group

Then each rank owns roughly:

$$
\frac{E}{P}
$$

experts.

If token `x_t` is routed to expert `e`, it must be sent to the rank that owns `e`.

So EP changes the computation pattern from:

```text
all tokens stay local
```

to:

```text
route token -> send to expert owner -> compute expert -> send output back
```

That is why the defining communication primitive of EP is usually **all-to-all**.

---

## 4. Step-by-Step Forward Pass

Consider a batch of token hidden states:

$$
H \in \mathbb{R}^{N \times d}
$$

where:

* `N` = number of tokens currently entering the MoE layer
* `d` = hidden size

### 4.1 Router Scores

A gating network produces expert scores:

$$
s_t = W_{\text{gate}} h_t
$$

and router probabilities:

$$
p_t = \operatorname{softmax}(s_t)
$$

Then top-`k` experts are selected:

$$
\mathcal{T}(h_t) = \operatorname{TopK}(p_t, k)
$$

### 4.2 Dispatch

Each token is assigned to its selected experts.
If expert `e` lives on rank `r(e)`, the token must be sent to that rank.

This produces a grouped token dispatch:

```text
Rank 0 sends some tokens to rank 1, some to rank 2, ...
Rank 1 sends some tokens to rank 0, some to rank 3, ...
...
```

This is why EP usually uses **all-to-all** rather than all-reduce.

### 4.3 Local Expert Compute

After dispatch, each rank runs its own local experts:

$$
z_{t,e} = E_e(h_t)
$$

for the tokens that arrived for expert `e`.

### 4.4 Combine Outputs

The expert outputs are sent back and combined with router weights:

$$
y_t
=
\sum_{e \in \mathcal{T}(h_t)}
p_{t,e}\,z_{t,e}
$$

So a full EP forward usually has:

1. routing
2. all-to-all dispatch
3. local expert MLP compute
4. all-to-all return
5. weighted combine

---

## 5. Diagram

```text
Tokens
  |
  v
Router / Top-K
  |
  v
All-to-All Dispatch
  |
  +--> Rank 0 local experts
  +--> Rank 1 local experts
  +--> Rank 2 local experts
  +--> ...
  |
  v
All-to-All Return
  |
  v
Weighted Combine
  |
  v
Final MoE output
```

If the model also has **shared experts** or a dense residual MLP branch, that branch is computed locally and added afterward.

---

## 6. Why EP Is Different from TP

This is an easy place to get confused.

### Tensor Parallelism

TP splits **one dense tensor** across GPUs.

Example:

* one large linear weight matrix is sharded
* every token still conceptually uses the whole layer

### Expert Parallelism

EP splits **different experts** across GPUs.

Example:

* expert 0, 1 on rank 0
* expert 2, 3 on rank 1
* expert 4, 5 on rank 2

and each token only visits the selected experts.

So:

* **TP** parallelizes dense linear algebra
* **EP** parallelizes sparse routing structure

---

## 7. Why All-to-All Is the Key EP Communication

With TP, communication is usually:

* all-reduce
* all-gather
* reduce-scatter

With EP, the difficult step is different:

* each rank must send **different subsets of tokens to different ranks**

That pattern is naturally:

* **all-to-all**

This is also why EP can become communication-heavy:

* token counts per expert are data-dependent
* token traffic is irregular
* load can become imbalanced

So EP is often bottlenecked less by math and more by routing and token movement.

---

## 8. Load Balancing Is the Central EP Problem

EP is powerful, but it has one major weakness:

> the router may send too many tokens to a few popular experts.

Then:

* those expert-owning ranks become overloaded
* other ranks sit partially idle
* communication buffers become skewed
* throughput drops

This is why MoE training almost always needs some form of load balancing.

---

## 9. Common Load-Balancing Strategies

### 9.1 Auxiliary Load-Balancing Loss

Many classic MoE systems add a router regularizer that encourages more even expert usage.

Conceptually:

$$
\mathcal{L}
=
\mathcal{L}_{\text{task}}
+
\lambda \mathcal{L}_{\text{balance}}
$$

where the balance term penalizes collapsed routing.

### 9.2 Capacity Factor

Some systems limit how many tokens an expert can accept in one step.

If too many tokens are routed to one expert:

* extra tokens may be dropped
* or rerouted
* or padded into a bounded capacity layout

This improves systems stability, though it can complicate training semantics.

### 9.3 Better Routing / Grouped Routing

Modern systems often improve the top-`k` routing procedure itself so that dispatch is more stable or hardware-friendly.

### 9.4 Redundant Experts / EPLB

DeepSeek-style systems discuss **EPLB**:

* Expert Parallel Load Balancing

The intuition is:

* if some experts become hotspots
* create redundant physical copies
* let tokens route to a less-loaded copy of the same logical expert

This is a systems-level answer to routing imbalance.

---

## 10. EP with Redundant Physical Experts

This idea is important enough to separate clearly.

Suppose the model defines:

* `E_logical` logical experts

but the runtime stores:

$$
E_{\text{physical}}
=
E_{\text{logical}} + E_{\text{redundant}}
$$

Then the router or dispatcher can choose among multiple physical copies for the same logical computation.

This helps because:

* the logical model structure stays the same
* hot experts no longer map to only one physical location

So EP load balancing is not only a machine learning issue.
It is also a **placement and dispatch issue**.

---

## 11. Backward Pass Under EP

Backward also follows the dispatch pattern.

If a token visited expert `e` on some remote rank in forward, then in backward:

* its gradient must flow back through that expert
* the corresponding expert parameter gradients are computed on that expert's owning rank

So EP backward usually also involves:

* token grouping
* all-to-all communication
* local gradient computation per expert

The overall pattern is still:

> route by expert ownership, compute locally, then move data back.

---

## 12. Memory and Compute Scaling

If experts are evenly distributed across `P` ranks, then expert parameter memory per rank is roughly:

$$
\frac{1}{P}
$$

of the full routed-expert parameter pool.

This is the main reason EP matters:

* large MoE models may have enormous total expert parameters
* but each token only uses a few experts
* EP lets us store and compute only the needed expert subset per rank

So EP gives:

* large parameter capacity
* sparse per-token compute
* manageable per-rank memory

This is why MoE can achieve very large total parameter counts without dense FLOPs scaling the same way.

---

## 13. EP in Practice: Dense Parts Still Need Other Parallelism

A real MoE model is not "only experts".

It still has:

* attention layers
* shared MLP branches
* embeddings
* output heads

Those parts are usually parallelized with other methods:

* **TP** for dense layers
* **PP** for deep models
* **DP** for throughput

So a realistic MoE training mesh is often:

```text
world size = DP x PP x TP x EP
```

This is one reason large-model systems engineering becomes complicated very quickly.

---

## 14. EP vs PP

These two solve very different problems.

### Pipeline Parallelism

* splits layers by depth
* each token still passes through the same sequence of stages

### Expert Parallelism

* splits experts inside MoE layers
* tokens dynamically visit different experts depending on routing

So:

* **PP** is structurally static
* **EP** is data-dependent and dynamic

That dynamic routing is why EP needs more complex dispatch logic.

---

## 15. Benefits

### Fits huge MoE expert pools

Without EP, storing all experts on every rank would be prohibitively expensive.

### Matches sparse computation

Only the selected experts run for each token.

### Enables large total parameter count

MoE can scale model capacity much more aggressively when EP is available.

---

## 16. Trade-offs

### All-to-all communication is expensive

Unlike dense TP collectives, token routing traffic is irregular and data-dependent.

### Load imbalance can dominate runtime

A few overloaded experts can hurt the whole step.

### Systems complexity is much higher

Token dispatch, padding, sorting, reordering, combining, and placement all matter.

### Small MoE may not benefit enough

If expert count is small, the extra routing complexity may not be worth it.

---

## 17. EP vs No EP in an MoE Model

| Aspect | No EP | EP |
|:---|:---|:---|
| Expert storage | all experts replicated | experts sharded across ranks |
| Token movement | mostly local | all-to-all dispatch |
| Memory per rank | high | lower |
| Runtime complexity | simpler | much more complex |
| Best use case | small MoE | large-scale sparse MoE |

---

## 18. Key Takeaways

1. **EP shards MoE experts across GPUs rather than replicating them everywhere**
2. **Tokens are routed to expert-owning ranks, so all-to-all is the core communication primitive**
3. **The forward path is: route -> dispatch -> local expert compute -> return -> combine**
4. **The biggest engineering challenge in EP is load balancing**
5. **Large MoE systems usually combine EP with TP, PP, and DP rather than using EP alone**
