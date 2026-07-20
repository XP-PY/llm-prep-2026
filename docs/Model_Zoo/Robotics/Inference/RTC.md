# [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)

## Convenient Links

* [Paper (arXiv)](https://arxiv.org/abs/2506.07339)
* [Project Page / Videos](https://pi.website/research/real_time_chunking)
* [Simulation Code](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
* [pi0 note in this repo](../Policies/Pi_0.md)
* [Diffusion Policy note in this repo](../Policies/Diffusion_Policy.md)
* [ACT / ALOHA note in this repo](../Policies/ACT.md)
* [SmolVLA note in this repo](../Policies/SmolVLA.md)

## 1. One-Sentence Summary

Real-Time Chunking (RTC) is an **inference-time algorithm** for diffusion- or flow-based action-chunking policies that runs action generation asynchronously while the robot is executing the current chunk, freezes actions that are already guaranteed to execute, and inpaints the rest of the next chunk to preserve cross-chunk continuity without retraining the policy.

## 2. Why RTC Matters

Action chunking is now common in robot policies:

```text
observation
-> policy
-> future action chunk
-> execute several actions
-> replan
```

It improves temporal consistency, but it does not remove inference latency.

For large VLAs, inference may take longer than the robot controller period. The paper gives two examples:

* pi0 spends `46 ms` on KV-cache prefill alone on an RTX 4090, before denoising
* optimized OpenVLA still reports about `321 ms` latency on an A100

For a `50 Hz` robot controller:

$$
\Delta t = 20 \text{ ms}
$$

So a policy taking `80-300 ms` cannot synchronously respond at every control tick.

Naive options are bad:

* **synchronous inference** pauses between chunks
* **naive async inference** switches to the new chunk when ready, causing discontinuities
* **temporal ensembling** averages actions from different modes, which can create invalid actions

RTC's core idea is:

> Generate the next chunk while executing the current one, but force the new chunk to be compatible with the old chunk through flow-based inpainting.

## 3. Problem Setup

An action-chunking policy is written as:

$$
\pi(A_t \mid o_t)
$$

where:

$$
A_t =
\left[
a_t,\,
a_{t+1},\,
\dots,\,
a_{t+H-1}
\right]
$$

Here:

* $H$ is the prediction horizon
* $o_t$ is the observation at controller timestep `t`
* $a_t$ is a low-level robot action

The policy predicts `H` future actions, but execution usually consumes only:

$$
s \le H
$$

actions before replanning. The paper calls `s` the **execution horizon**.

The basic chunking tradeoff is:

| Execution horizon | Behavior |
| :---------------- | :------- |
| large `s` | smoother open-loop execution, less reactive |
| small `s` | more reactive, but more likely to jump between modes |

RTC is designed to keep the reactivity of shorter horizons while avoiding cross-chunk jumps.

## 4. Flow Policy Background

RTC is described for conditional flow-matching action policies, though the paper notes that diffusion policies can be converted to flow policies at inference time.

Start from random noise:

$$
A_t^0 \sim \mathcal{N}(0, I)
$$

Then integrate the learned velocity field:

$$
A_t^{\tau + \frac{1}{n}}
=
A_t^\tau
+
\frac{1}{n}
v_\pi(A_t^\tau, o_t, \tau)
$$

where:

* $\tau \in [0, 1)$ is the flow timestep
* $n$ is the number of denoising / integration steps
* $v_\pi$ is the learned policy velocity field

After `n` steps, the model obtains a clean action chunk:

$$
A_t^1
$$

The real-time problem is not how to train this policy. RTC assumes the policy already exists.

The problem is how to execute it when:

$$
\delta > \Delta t
$$

where $\delta$ is policy inference time and $\Delta t$ is the controller period.

## 5. Inference Delay

The paper defines the inference delay:

$$
d
:=
\left\lfloor
\frac{\delta}{\Delta t}
\right\rfloor
$$

This is the number of controller timesteps that pass between observing $o_t$ and receiving the generated chunk $A_t$.

For example, if:

$$
\Delta t = 20\text{ ms},
\qquad
\delta = 97\text{ ms}
$$

then:

$$
d \approx 4
$$

In the real-robot experiments, remote inference and network latency make the effective delay larger, around:

| Setting | Approximate delay |
| :------ | ----------------: |
| base LAN setup | $d \approx 6$ |
| `+100 ms` injected latency | $d \approx 11$ |
| `+200 ms` injected latency | $d \approx 16$ |

## 6. Why Naive Async Fails

First, define the action notation:

$$
a_{n|m}
$$

Here, $n$ is the timestep when the action should be executed, and $m$ is the timestep of the observation used to predict it. Therefore:

$$
A_m
=
\left[
a_{m|m},\,
a_{m+1|m},\,
\dots,\,
a_{m+H-1|m}
\right]
$$

The position of $a_{n|m}$ inside $A_m$ is $n-m$; $n$ is an absolute controller timestep, not a chunk-local index.

Now suppose the robot starts by executing a chunk predicted from $o_0$:

$$
A_0
=
\left[
a_{0|0},\,
a_{1|0},\,
\dots
\right]
$$

The controller plans to switch to a new chunk at timestep $s$. Because inference takes $d$ timesteps, it must start inference at timestep $s-d$ using the observation $o_{s-d}$:

$$
(s-d)+d=s
$$

The asynchronous timeline is:

| Time | Robot controller | Background policy inference |
| :--- | :--------------- | :-------------------------- |
| $0$ to $s-d-1$ | executes actions from $A_0$ | idle |
| $s-d$ | executes $a_{s-d\mid 0}$ | captures $o_{s-d}$ and starts generating $A_{s-d}$ |
| $s-d+1$ to $s-1$ | continues executing $A_0$ | continues generating $A_{s-d}$ |
| $s$ | receives and switches to $A_{s-d}$ | inference finishes |

The returned chunk covers actions beginning at timestep $s-d$:

$$
A_{s-d}
=
\left[
a_{s-d|s-d},\,
\dots,\,
a_{s-1|s-d},\,
a_{s|s-d},\,
\dots
\right]
$$

By the time this chunk arrives at timestep $s$, its first $d$ predictions correspond to timesteps that have already passed:

$$
a_{s-d|s-d},\,
\dots,\,
a_{s-1|s-d}
$$

Naive asynchronous execution discards these stale actions and immediately executes the first still-current prediction, $a_{s|s-d}$.

The problem is that $A_{s-d}$ was sampled independently. It observed the state at $s-d$, but it was not constrained to continue the actions from $A_0$ that the robot executed while inference was running. The policy can therefore choose a different valid strategy, especially when its action distribution is multimodal.

At the handoff, the controller abruptly changes from:

$$
a_{s-1|0}
\quad\longrightarrow\quad
a_{s|s-d}
$$

Nothing in naive asynchronous sampling guarantees that these consecutive actions are compatible. The result can be a large action jump, acceleration, or jerk.

Example:

```text
at time s-d: old chunk is moving above an obstacle
new inference independently chooses the valid "move below" mode
until time s: robot keeps following the old "move above" mode
at time s: naive async switches to the middle of the "move below" chunk
result: an abrupt, potentially out-of-distribution correction
```

Temporal ensembling can make the numerical transition smoother, but it does not solve the mode mismatch: averaging an "above" action and a "below" action may produce an invalid action aimed directly at the obstacle. RTC instead generates the new chunk while explicitly constraining it to remain compatible with the old chunk.

## 7. RTC as Inpainting

**Inpainting** means completing a partially specified sample. Known entries provide context or constraints, and the missing entries are generated so that the completed sample is coherent. Image inpainting, for example, preserves visible pixels while generating a missing region that fits the surrounding image.

RTC applies this idea along the time dimension of an action chunk. It treats actions from the previous chunk as the known context and generates a new chunk that continues from them smoothly.

### 7.1 Aligning the Old and New Chunks

For clarity, follow Figure 3 and set the timestep when new inference starts to $0$. The previous inference call was made $s$ timesteps earlier, at timestep $-s$.

That previous call used $o_{-s}$ to generate:

$$
A_{-s}
=
\left[
a_{-s|-s},\,
\dots,\,
a_{H-s-1|-s}
\right]
$$

The new inference call uses $o_0$ and generates:

$$
A_0
=
\left[
a_{0|0},\,
\dots,\,
a_{H-1|0}
\right]
$$

Both chunks cover timesteps $0$ through $H-s-1$. Their overlapping predictions can therefore be aligned by intended execution time:

$$
\underbrace{
\left[
a_{0|-s},\,
\dots,\,
a_{H-s-1|-s}
\right]
}_{\text{remaining actions from the previous chunk}}
\quad\leftrightarrow\quad
\underbrace{
\left[
a_{0|0},\,
\dots,\,
a_{H-s-1|0}
\right]
}_{\text{overlapping part of the new chunk}}
$$

The previous chunk provides no reference for the last $s$ actions of $A_0$, because they extend beyond its prediction horizon.

### 7.2 Three Action Regions

Inference takes $d$ timesteps, so the new chunk is not available until timestep $d$. This divides $A_0$ into three regions:

| New-chunk index $i$ | Region | What happens |
| :------------------ | :----- | :----------- |
| $0 \le i < d$ | frozen prefix | The robot executes $a_{i\mid -s}$ while $A_0$ is still being generated, so these actions can no longer be changed. RTC constrains $a_{i\mid 0}$ to match them. |
| $d \le i < H-s$ | editable overlap | The new chunk will be ready before these actions are needed. RTC may update them using $o_0$, but guides them toward the previous chunk to preserve its strategy. |
| $H-s \le i < H$ | new tail | These timesteps are beyond the end of $A_{-s}$. There is no old action to match, so the policy generates them freely. |

In Figure 3, $H=16$, $s=5$, and $d=4$. Thus, $a_0$ through $a_3$ form the frozen prefix, $a_4$ through $a_{10}$ form the editable overlap, and $a_{11}$ through $a_{15}$ form the new tail.

![RTC](../../../../assets/RTC.png)

The frozen prefix may appear unnecessary because it is already stale when inference finishes. However, it records the actions the robot actually executed during inference. Conditioning the rest of the generated trajectory on this prefix makes the usable suffix begin as a coherent continuation of the robot's real motion.

### 7.3 Hard and Soft Inpainting

A basic, hard-masked version of inpainting would enforce only:

$$
a_{i|0} \approx a_{i|-s},
\qquad 0 \le i < d
$$

and generate every later action freely. This can still allow the new chunk to switch strategies immediately after the frozen prefix.

RTC therefore also uses the editable overlap as a **soft constraint**. The guidance is strongest near the frozen prefix, decreases for actions farther into the future, and becomes zero in the new tail:

```text
frozen prefix       editable overlap             new tail
must match     ->   gradually weaker match   ->  generate freely
0 ... d-1           d ... H-s-1                  H-s ... H-1
```

Inpainting does not mean copying the previous chunk. It means sampling a complete new chunk from the policy, conditioned on the latest observation, while using the previous chunk as temporal context. This gives the policy room to react to $o_0$ without abruptly changing the strategy already being executed.

## 8. Guidance-Based Inpainting

RTC performs inpainting by adding **gradient guidance** to the velocity predicted by the original flow policy. The base velocity keeps generation consistent with the learned action distribution, while the guidance term encourages the generated chunk to agree with the previous chunk where they overlap. This is an inference-time modification; the policy is not retrained.

Two different time variables appear in this section:

* $t$ is a controller timestep, such as the moment when observation $o_t$ is captured.
* $\tau \in [0,1]$ is a flow-sampling timestep. Sampling starts from noise at $\tau=0$ and ends with a clean action chunk at $\tau=1$.

### 8.1 Constructing the Inpainting Target

Suppose a new inference call begins at controller timestep $t$, and the previous call began at $t-s$. Following the alignment from Chapter 7, the previous chunk supplies target actions for the overlapping interval:

$$
Y_i
=
\begin{cases}
a_{t+i\mid t-s}, & 0 \le i < H-s
\\
0, & H-s \le i < H
\end{cases}
$$

Thus, $Y$ is the remaining part of the previous chunk, aligned with the new chunk and right-padded to length $H$. The padding value is irrelevant because the corresponding mask weights are zero.

Let $W_i$ indicate how strongly position $i$ should match $Y_i$:

| Region | Weight | Role |
| :----- | :----- | :--- |
| frozen prefix, $0 \le i < d$ | $W_i=1$ | match actions guaranteed to execute during inference |
| editable overlap, $d \le i < H-s$ | $0<W_i<1$ | encourage a smooth continuation without completely preventing correction |
| new tail, $H-s \le i < H$ | $W_i=0$ | generate freely because the previous chunk has no corresponding action |

If each robot action has dimension $D$, then $Y$ and $A_t^\tau$ contain $H \times D$ values. The equations treat them as flattened vectors and apply each timestep weight $W_i$ to all $D$ components of action $i$.

Chapter 9 defines the exact decay used for the editable-overlap weights.

### 8.2 Estimating the Clean Chunk

At flow timestep $\tau$, let $A_t^\tau$ be the current noisy sample for the new chunk. The base policy predicts its velocity:

$$
v_\tau
=
v(A_t^\tau,o_t,\tau)
$$

RTC uses this velocity to estimate where the sample would end at $\tau=1$ if that velocity remained constant:

$$
\widehat{A_t^1}
=
A_t^\tau
+
(1-\tau)v_\tau
$$

$\widehat{A_t^1}$ is an estimate of the final clean action chunk, not the next integration state. Comparing the target with this clean estimate is more meaningful than comparing it with the noisy sample $A_t^\tau$ directly.

### 8.3 Measuring Inpainting Error

Define a weighted reconstruction loss:

$$
\mathcal{L}_{\mathrm{inp}}
=
\frac{1}{2}
\left(
Y-\widehat{A_t^1}
\right)^\top
\operatorname{diag}(W)
\left(
Y-\widehat{A_t^1}
\right)
$$

Only positions with nonzero $W_i$ contribute to this loss. A frozen-prefix error contributes fully, an editable-overlap error contributes partially, and a new-tail error contributes nothing.

RTC then differentiates this loss with respect to the current noisy sample. Let:

$$
J_\tau
=
\frac{\partial \widehat{A_t^1}}
{\partial A_t^\tau}
$$

The direction that reduces the inpainting error is:

$$
g_\tau
=
-\nabla_{A_t^\tau}\mathcal{L}_{\mathrm{inp}}
=
J_\tau^\top
\operatorname{diag}(W)
\left(
Y-\widehat{A_t^1}
\right)
$$

This is the vector-Jacobian product in the paper's PiGDM-style guidance equation. It can be computed with reverse-mode automatic differentiation; RTC does not construct the full Jacobian matrix.

### 8.4 Correcting the Flow Velocity

The guided velocity combines the policy's original prediction with the inpainting correction:

$$
v_{\Pi\mathrm{GDM}}(A_t^\tau,o_t,\tau)
=
v_\tau
+
\lambda_\tau g_\tau
$$

The sampler then uses this corrected velocity in its ordinary integration step:

$$
A_t^{\tau+\Delta\tau}
=
A_t^\tau
+
\Delta\tau\,
v_{\Pi\mathrm{GDM}}(A_t^\tau,o_t,\tau)
$$

The guidance scale is:

$$
\lambda_\tau
=
\min
\left(
\beta,\,
\frac{1-\tau}{\tau r_\tau^2}
\right),
\qquad
r_\tau^2
=
\frac{(1-\tau)^2}
{\tau^2+(1-\tau)^2}
$$

The $\tau$-dependent factor adjusts the correction over the flow trajectory. The clipping constant $\beta$ prevents excessively large guidance updates, which are especially unstable when robot policies use only a few denoising steps. The paper uses:

$$
\beta=5
$$

The complete logic at every flow step is therefore:

```text
predict the base velocity
estimate the final clean chunk
measure its weighted mismatch with the aligned previous chunk
backpropagate that mismatch to the current noisy sample
add the clipped correction to the base velocity
integrate one step
```

The correction does not directly overwrite actions. It steers the entire denoising trajectory toward a sample that is both likely under the policy and compatible with the robot's recent motion.

## 9. Soft Masking

Chapter 7 divided the new chunk into frozen, editable-overlap, and new-tail regions. Chapter 8 used $W$ to weight their inpainting errors. RTC defines those weights with the following schedule:

$$
W_i =
\begin{cases}
1,
& i < d
\\
c_i \frac{e^{c_i} - 1}{e - 1},
& d \le i < H - s
\\
0,
& i \ge H - s
\end{cases}
$$

where:

$$
c_i
=
\frac{H - s - i}
{H - s - d + 1},
\qquad
i \in \{0,\dots,H-1\}
$$

For the editable overlap, $c_i$ measures how far position $i$ is from the end of the overlap. As $i$ moves farther into the future, $c_i$ and therefore $W_i$ decrease toward zero.

Using the Figure 3 values $H=16$, $s=5$, and $d=4$:

$$
W_4 \approx 0.71,
\qquad
W_7 \approx 0.19,
\qquad
W_{10} \approx 0.01
$$

The schedule has two important effects:

* The first $d$ positions receive full guidance because those actions will actually execute before inference finishes.
* The rest of the overlap acts as a gradually weakening strategy prior rather than a hard trajectory constraint.

$W$ is not used to average old and new actions. It weights the reconstruction error before backpropagation, so the final result remains one jointly generated action chunk. A larger delay $d$ lengthens the frozen prefix, while a larger execution horizon $s$ shortens the available overlap and enlarges the freely generated tail.

## 10. Full RTC Runtime Loop

The inpainting equations describe how to generate one chunk. RTC also needs a scheduler that keeps the controller supplied with actions while generation runs in the background.

### 10.1 Shared Runtime State

The controller and inference thread share:

* the active action chunk and its current read index
* the latest observation
* the number of controller steps since inference last started
* a bounded buffer $\mathcal{B}$ of recently observed inference delays

Access to the active chunk and counters must be synchronized so that replacing a chunk cannot race with reading the next action.

RTC predicts the next delay conservatively from the recent buffer:

$$
\widehat d
=
\max \mathcal{B}
$$

It then chooses the next execution horizon:

$$
s
=
\max
\left(
s_{\min},\,
\widehat d
\right)
$$

Here, $s_{\min}$ controls the minimum amount of open-loop execution, while $\widehat d$ prevents the inference thread from scheduling requests faster than it can complete them. The paper generally writes $d$ for the delay used by the scheduler; $\widehat d$ makes explicit that the next delay must be estimated before inference runs.

### 10.2 Controller Thread

Every $\Delta t$, the real-time controller performs only short buffer operations:

```text
read the next action from the active chunk
advance the chunk index
store the newest observation
increment the steps-since-inference counter
wake the inference thread if its start condition is satisfied
send the action to the robot
```

The controller never waits for model inference. While the model is working, it continues consuming actions from the current chunk.

### 10.3 Background Inference Thread

When $s$ controller steps have passed since the previous inference start, the background thread:

```text
1. snapshot the latest observation and the unconsumed old chunk
2. estimate the next delay d_hat from recent measurements
3. align and pad the old chunk to construct Y
4. construct W from H, s, and d_hat
5. run guided flow sampling to generate the new chunk
6. measure the actual delay d_actual
7. atomically replace the active chunk
8. set its read index to d_actual, skipping actions whose times passed
9. record d_actual in the delay buffer
```

The atomic swap is the only point at which the controller changes chunks. The first $d_{\mathrm{actual}}$ entries of the new chunk are not executed; their purpose was to anchor generation to the motion that occurred during inference.

### 10.4 Feasible Timing

RTC requires:

$$
d \le s \le H - d
$$

The two bounds have different roles:

* $d \le s$ ensures one inference call finishes before the next call is scheduled, preventing an inference backlog.
* $s \le H-d$ is equivalent to $H-s \ge d$, ensuring the previous chunk still contains at least $d$ actions to execute while the new chunk is generated.

In practice these conditions use the conservative estimate $\widehat d$. If the actual delay exceeds the estimate, RTC records the larger value so subsequent cycles reserve a longer delay window.

## 11. Guided Inference Implementation

Chapters 8 and 9 provide the equations. The following pseudocode shows how they fit into one guided sampler without restating the derivation:

```text
function GuidedInference(observation o, target Y, weights W,
                         denoising_steps n, guidance_clip beta):
    A = sample_standard_gaussian()
    delta_tau = 1 / n

    for k in 0, ..., n - 1:
        tau = k / n
        enable_gradient(A)

        v_base = policy_velocity(A, o, tau)
        A_clean_hat = A + (1 - tau) * v_base

        loss = 0.5 * sum(W * (Y - A_clean_hat)^2)
        guidance = -gradient(loss, with_respect_to=A)

        r_squared = (1 - tau)^2 / (tau^2 + (1 - tau)^2)
        if tau == 0:
            lambda = beta
        else:
            lambda = min(beta, (1 - tau) / (tau * r_squared))

        v_guided = v_base + lambda * guidance
        A = stop_gradient(A + delta_tau * v_guided)

    return A
```

At $\tau=0$, the unclipped guidance expression is singular; the clipping rule makes the effective value $\beta$. The timestep weights $W_i$ are broadcast over all dimensions of action $i$.

Only gradients with respect to the noisy action sample $A$ are needed. RTC does not compute parameter gradients and does not update the policy. Nevertheless, each denoising step now requires both a model forward pass and a reverse-mode gradient computation, explaining RTC's higher latency relative to vanilla flow sampling.

## 12. Method Summary

RTC maintains one central invariant:

> When a new chunk becomes executable, its usable suffix has been generated as a continuation of the actions the robot actually executed during inference.

The complete data flow is:

```text
old chunk + latest observation + delay estimate
    -> aligned target Y and soft weights W
    -> guided flow sampler
    -> new coherent chunk
    -> skip stale prefix and atomically swap
```

| Component | What RTC changes |
| :-------- | :--------------- |
| policy training and weights | nothing |
| flow sampling | adds a weighted input-gradient correction at every denoising step |
| temporal context | aligns the previous chunk with the new chunk and uses it as guidance |
| execution | generates asynchronously and swaps chunks without pausing the controller |
| delay handling | predicts delay from recent measurements and freezes the corresponding prefix |
| compute cost | adds one reverse-mode gradient computation per denoising step |

RTC therefore preserves reactivity to the latest observation without either independently jumping between chunks or averaging incompatible action modes. It applies directly to flow policies and to diffusion policies that can be converted to an equivalent flow sampler at inference time.

## 13. Evaluation at a Glance

The experiments test four claims:

1. RTC remains effective as inference delay increases.
2. Guided sampling preserves continuity better than switching or averaging chunks.
3. Softly weighting the full overlap works better than freezing only the unavoidable prefix.
4. These improvements increase real-robot task throughput, not just trajectory smoothness.

The paper uses two complementary testbeds:

| | Kinetix simulation | Real robot |
| :--- | :--- | :--- |
| Main purpose | controlled stress test for delay and multimodality | measure task speed and completion under realistic VLA latency |
| Tasks | `12` dynamic manipulation and locomotion tasks | `6` bimanual manipulation tasks |
| Base policy | action-chunking flow policy | pi0.5 |
| Prediction horizon | $H=8$ | $H=50$ |
| Delay conditions | $d=0$ to $4$ | approximately $d=6$, $11$, and $16$ |
| Evaluation scale | `2048` rollouts per data point | `480` episodes and `28` robot-hours |

Simulation isolates why RTC works; the real-world study tests whether the same mechanism improves useful robot behavior.

## 14. Simulation: Controlled Stress Test

### 14.1 Why Kinetix

Many manipulation benchmarks are quasi-static: a robot can stop while inference runs without substantially changing the task. Kinetix instead uses force-based dynamics, noisy actions, and motions such as throwing, catching, balancing, landing, and locomotion. The environment continues evolving while the policy is computing, so delay cannot be hidden by holding position.

### 14.2 Training and Evaluation

| Item | Setting |
| :--- | :------ |
| Environments | `12` Kinetix tasks |
| Demonstration sources | `6` RPO experts per environment |
| Dataset | `1M` transitions per environment |
| Learned policy | 4-layer MLP-Mixer flow policy |
| Training | `32` epochs |
| Prediction horizon | $H=8$ |
| Evaluation | `2048` rollouts per data point |
| Simulated inference delay | $d=0$ to $4$ |

### 14.3 Compared Methods

| Method | How it handles chunk boundaries |
| :----- | :------------------------------ |
| Naive async | replaces the old chunk as soon as the new one arrives |
| Temporal ensembling | averages all predictions for the same execution timestep |
| BID | rejection-samples chunks to find a continuation compatible with the old chunk |
| Hard-mask RTC | guides only the first $d$ frozen actions |
| Soft-mask RTC | uses full guidance on the frozen prefix and decaying guidance over the remaining overlap |

### 14.4 What the Simulation Shows

| Observation | Interpretation |
| :---------- | :------------- |
| RTC degrades least as $d$ increases | explicitly modeling the delay is important in dynamic environments |
| RTC outperforms BID, with a larger gap at higher delay | guided inpainting is more effective than selecting among independently sampled chunks |
| BID uses batches of `64` candidate chunks | its continuity comes with substantially higher sampling cost |
| Temporal ensembling performs poorly even at $d=0$ | an average of two valid action modes may itself be invalid |
| Soft masking outperforms hard masking | a short frozen prefix may not contain enough information to preserve the old strategy |
| RTC improves as the execution horizon becomes shorter | continuity lets the policy replan more often and benefit from closed-loop corrections |

The central result is not simply that RTC makes actions numerically smoother. It keeps the new chunk within a coherent mode of the learned action distribution while still allowing future actions to react to the latest observation.

## 15. Real Robot: Protocol

### 15.1 System and Delay Conditions

The real-world study uses pi0.5 on a bimanual platform with two 6-DoF arms and parallel-jaw grippers.

| Setting | Value |
| :------ | :---- |
| Prediction horizon | $H=50$ actions |
| Controller | `50 Hz`, or $\Delta t=20\text{ ms}$ |
| Flow steps | $n=5$ |
| Remote inference overhead | `10-20 ms` over LAN |
| Normal RTC delay | $d\approx6$ |
| Added latency | `+100 ms` and `+200 ms` |
| Resulting delays | $d\approx11$ and $d\approx16$ |

The largest delay consumes more than `30%` of the prediction horizon before a new chunk becomes available.

### 15.2 Tasks and Scoring

| Task | Substeps | Cutoff | Main challenge |
| :--- | -------: | -----: | :------------- |
| Light candle | `5` | `40 s` | precise, dynamic match striking and lighting |
| Plug ethernet | `6` | `120 s` | repeated cable grasping, alignment, and insertion |
| Make bed, mobile | `3` | `200 s` | mobile manipulation of a blanket and pillows |
| Shirt folding | `1` | `300 s` | long-horizon deformable-object manipulation |
| Batch folding | `4` | `300 s` | flattening, folding, and stacking varied clothing |
| Dishes in sink, mobile | `8` | `300 s` | repeated mobile pick-and-place with varied objects |

Each task-method setting is evaluated for `10` trials. Episodes receive partial credit for completed substeps, and the completion time of each substep is recorded. The full study contains `480` episodes and about `28` hours of robot execution.

### 15.3 Methods and Metric

| Method | Runtime behavior |
| :----- | :--------------- |
| Synchronous | executes $s=25$ actions, pauses for inference, then starts the next chunk |
| TE sparse | runs asynchronous inference with sparse overlap and temporal ensembling |
| TE dense | infers as often as possible and averages several overlapping chunks |
| RTC | runs asynchronous guided inpainting with a soft overlap mask |

BID is omitted from the real robot because it is weaker in simulation and expensive. With pi0.5 and a batch size of `16`, its latency is approximately `2.3x` that of RTC.

The primary metric is average throughput:

$$
\text{throughput}
=
\frac{\text{proportion of task completed}}
{\text{episode duration}}
$$

Throughput rewards both completing more of the task and completing it sooner. It is more informative here than final success alone because most tasks allow retries until their time limit.

## 16. Real Robot: Results

| Question | Result |
| :------- | :----- |
| Which method has the highest throughput? | RTC at every tested delay; its advantage is statistically significant at `+100 ms` and `+200 ms`. |
| How does added delay affect RTC? | RTC shows no measurable degradation across the injected-delay conditions. |
| How does delay affect synchronous execution? | Throughput falls roughly linearly because every additional millisecond becomes robot idle time. |
| Do temporal-ensembling baselines remain stable? | No. At `+100 ms` and `+200 ms`, oscillations trigger the robot's protective stop. |
| Is RTC better only because it removes pauses? | No. RTC completes tasks faster even when synchronous inference pauses are removed from the timing analysis. |
| Where are the task-level gains strongest? | Light candle, the most precision-sensitive task, and make bed, the hardest task overall. |

The match-lighting result summarizes the practical effect: RTC succeeds with more than `300 ms` of inference delay, performs the same motion about `20%` faster than synchronous execution, and produces smoother motion than temporal ensembling.

The pause-adjusted comparison is important. It indicates that RTC improves control quality by reducing mistakes and retries, rather than merely overlapping computation with execution.

## 17. Compute Cost and Settings

### 17.1 Latency Breakdown

On an RTX 4090, the pi0.5 latency is:

| Component | Vanilla pi0.5 | pi0.5 with RTC |
| :-------- | ------------: | -------------: |
| SigLIP image encoding | `18 ms` | `18 ms` |
| Gemma 2B prefill | `44 ms` | `44 ms` |
| Five flow steps | `14 ms` | `35 ms` |
| Total | `76 ms` | `97 ms` |

RTC adds `21 ms` to chunk generation. The image and language paths are unchanged; the extra cost comes from the vector-Jacobian product at every flow step.

RTC therefore does not reduce model latency. It accepts slightly slower chunk generation in exchange for continuous robot execution and compatible chunk transitions.

### 17.2 Paper Settings

| Symbol | Meaning | Simulation | Real robot |
| :----- | :------ | :--------- | :--------- |
| $n$ | flow steps | `5` | `5` |
| $H$ | prediction horizon | `8` | `50` |
| $s_{\min}$ | minimum execution horizon | not specified | `25` |
| $\beta$ | maximum guidance weight | `5` | `5` |
| $b$ | delay-history buffer size | not used | `10` |

The appendix compares several masks. Exponential decay performs best overall, with linear decay close behind. A Diffuser-style inpainting baseline improves over unguided generation but remains weaker than the gradient-guided RTC sampler.

## 18. When to Use RTC

| Method | Appropriate when | Main drawback |
| :----- | :--------------- | :------------ |
| Synchronous chunking | inference is fast enough or stopping does not change task dynamics | pauses scale directly with latency |
| Naive async | useful mainly as a diagnostic baseline | independent chunks can switch strategies abruptly |
| Temporal ensembling | overlapping predictions are approximately unimodal | averaging different modes can create invalid actions |
| BID | multiple candidate samples are affordable | high sampling cost and weaker simulated performance than RTC |
| Trained streaming or consistency policy | retraining or distillation is acceptable | requires a new training pipeline or model |
| RTC | an existing flow/diffusion chunk policy must run asynchronously without retraining | adds reverse-mode compute and needs sufficient chunk overlap |

RTC is a strong fit when all of the following hold:

* the policy already generates action chunks with a flow or diffusion process
* inference is slower than one controller period
* stopping or switching chunks changes task dynamics
* the prediction horizon leaves enough overlap to cover inference delay
* inference hardware can support backpropagation with respect to the action sample

Its main limitations are equally concrete:

* it does not make the underlying policy faster
* it adds a backward pass to every sampling step
* it does not directly apply to autoregressive or deterministic action heads
* continuity guidance becomes weak when little overlap remains
* the real-world evidence covers manipulation, not legged locomotion

The practical conclusion is concise: use asynchronous execution to remove waiting, and use RTC's guided sampling to make that asynchronous execution behaviorally coherent.
