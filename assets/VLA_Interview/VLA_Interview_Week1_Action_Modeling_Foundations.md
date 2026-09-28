# Week 1 — VLA Action Modeling Foundations

> Goal: rebuild the mathematical and implementation foundations behind modern VLA action generation for interview preparation.
>
> Main topics:
>
> **Behavior Cloning → Action Chunking → Diffusion Policy → Flow Matching / π0 → Autoregressive Action Tokens → FAST → Comparison & Interview Drills**
>
> Suggested workload: **15–18 hours**

---

## 0. What You Should Be Able to Do After This Week

By the end of Week 1, you should be able to answer the following **without opening notes**:

1. What exactly is Behavior Cloning optimizing?
2. Why does one-step regression often fail for robot control?
3. What is action chunking, and why do ACT, Diffusion Policy, and π0 all use it?
4. Write the DDPM forward-noising equation from memory.
5. Explain what the diffusion model predicts and why MSE on noise works.
6. Derive the basic Flow Matching target from a linear interpolation path.
7. Explain why π0 uses a target like $a-\epsilon$.
8. Explain why some papers instead write $\epsilon-a$.
9. Compare Diffusion and Flow Matching at training and inference time.
10. Explain continuous action generation vs autoregressive action-token generation.
11. Explain RT-2/OpenVLA-style discretization and FAST-style action tokenization.
12. Explain the trade-offs among:
    - direct regression
    - diffusion
    - flow matching
    - autoregressive action tokens
13. Given a VLA training batch, identify:
    - observation
    - proprioception
    - action chunk
    - noise
    - timestep
    - conditioning
    - prediction target
    - loss
14. Write a minimal Flow Matching training loop from memory.
15. Answer common interview follow-up questions rather than only reciting definitions.

---

# 1. Week Schedule

## Day 1 — Behavior Cloning + Action Chunking
**Estimated time: 2.5–3 h**

Study:

- Behavior Cloning objective
- MSE / L1 as supervised imitation objectives
- covariate shift
- multimodal action distributions
- action chunking
- ACT temporal ensembling
- prediction horizon vs execution horizon

Output:

- derive BC loss
- explain why action chunks reduce effective decision horizon
- explain why chunking does **not** fully solve covariate shift

---

## Day 2 — Diffusion Fundamentals
**Estimated time: 3 h**

Study:

- Gaussian noise
- DDPM forward process
- $\alpha_t,\beta_t,\bar\alpha_t$
- noise prediction
- conditional diffusion
- diffusion over actions rather than images
- Diffusion Policy training pipeline

Output:

- write DDPM forward equation without notes
- explain Diffusion Policy in 2 minutes
- sketch training/inference pseudocode

---

## Day 3 — Flow Matching Fundamentals
**Estimated time: 3 h**

Study:

- interpolation path
- velocity field
- conditional flow matching
- linear path derivation
- Euler integration
- π0 formulation
- sign / time-direction conventions

Output:

- derive $v^*=x_1-\epsilon$
- explain why WALL-style notation can produce $\epsilon-x_0$
- write a minimal Flow Matching implementation

---

## Day 4 — Diffusion vs Flow Matching
**Estimated time: 2–2.5 h**

Study:

- what is shared
- what differs
- noise prediction vs velocity prediction
- discrete denoising steps vs ODE integration
- inference cost
- multimodality
- conditioning
- action chunks

Output:

- one comparison table from memory
- answer 8 interview questions

---

## Day 5 — Autoregressive Actions + FAST
**Estimated time: 2.5–3 h**

Study:

- RT-2 action tokens
- OpenVLA action tokens
- continuous-to-discrete quantization
- cross-entropy objective
- autoregressive decoding
- FAST:
  - normalization
  - DCT
  - quantization
  - BPE

Output:

- explain why action tokenization lets VLMs reuse next-token prediction
- explain disadvantages of naive per-dimension bins
- explain why FAST compresses action trajectories better

---

## Day 6 — Integrated Review + Mock Interview
**Estimated time: 2–3 h**

Study:

- four policy families
- training batch anatomy
- implementation-level questions
- your RM65 / π0.5 project mapping

Output:

- 20-minute whiteboard explanation
- answer interview question bank at the end of this note
- mark all weak points with TODO

---

# 2. Behavior Cloning

## 2.1 Basic Setup

A robot demonstration dataset contains trajectories:

$$
\mathcal D=
\{
(o_t,a_t)
\}
$$

where:

- $o_t$: observation at time $t$
- $a_t$: demonstrated action

A policy models:

$$
\pi_\theta(a_t\mid o_t)
$$

Behavior Cloning treats policy learning as supervised learning.

The maximum-likelihood objective is:

$$
\theta^*
=
\arg\max_\theta
\sum_{(o,a)\in\mathcal D}
\log \pi_\theta(a\mid o)
$$

or equivalently:

$$
\mathcal L_{\mathrm{BC}}
=
-\mathbb E_{(o,a)\sim\mathcal D}
[
\log\pi_\theta(a\mid o)
]
$$

---

## 2.2 Why MSE Appears

Suppose the policy outputs the mean of a Gaussian:

$$
\pi_\theta(a\mid o)
=
\mathcal N(
a;
\mu_\theta(o),
\sigma^2 I
)
$$

Then:

$$
-\log \pi_\theta(a\mid o)
=
\frac{1}{2\sigma^2}
\|a-\mu_\theta(o)\|_2^2
+
C
$$

Therefore maximizing likelihood is equivalent to minimizing:

$$
\mathcal L_{\mathrm{MSE}}
=
\|a-\hat a\|_2^2
$$

when variance is fixed.

### Interview point

MSE is not arbitrary.

It corresponds to a **unimodal Gaussian assumption** over actions.

---

# 3. Why One-Step BC Is Often Not Enough

## 3.1 Covariate Shift

Training states come from the expert:

$$
s\sim d_{\pi_E}(s)
$$

but deployment states come from the learned policy:

$$
s\sim d_{\pi_\theta}(s)
$$

In general:

$$
d_{\pi_\theta}(s)
\neq
d_{\pi_E}(s)
$$

A small prediction error can change the next observation.

That next observation may be outside the demonstration distribution.

Then the error can compound.

### Typical interview answer

> Behavior Cloning is trained under the expert state distribution but deployed under its own induced state distribution. Small control errors can push the robot into out-of-distribution states, after which supervised imitation has no guarantee of recovery.

---

## 3.2 Multimodality

Suppose both actions are valid:

$$
a_1=-1,\qquad a_2=+1
$$

MSE regression may predict:

$$
\hat a=0
$$

even though $0$ is not a valid behavior.

For manipulation, this appears when:

- grasping from left vs right
- going around an obstacle from two sides
- choosing different valid approach poses
- human demonstrations vary in style

This motivates richer action distributions:

- CVAE
- GMM
- diffusion
- flow models
- discrete autoregressive tokens

---

# 4. Action Chunking

Instead of:

$$
\pi(a_t\mid o_t)
$$

predict:

$$
\pi(A_t\mid o_t)
$$

where:

$$
A_t=
[
a_t,
a_{t+1},
\dots,
a_{t+H-1}
]
$$

---

## 4.1 Why Action Chunking Helps

### 1. Temporal consistency

Nearby robot actions are strongly correlated.

Predicting them together allows the model to represent a coherent local motion.

### 2. Reduced effective decision horizon

For an episode of $T$ control steps, a one-step policy makes roughly:

$$
T
$$

high-level predictions.

A chunked policy may reduce the effective planning horizon toward:

$$
T/H
$$

although exact runtime behavior depends on how much of each chunk is executed.

### 3. Better modeling of manipulation primitives

A chunk can represent:

- reach
- grasp
- lift
- rotate
- insert

rather than treating each low-level timestep as an unrelated decision.

---

# 5. Prediction Horizon vs Execution Horizon

Suppose the model predicts:

$$
H_p=50
$$

actions.

But the robot executes only:

$$
H_e=10
$$

before observing again.

Then:

```text
observe
↓
predict 50 actions
↓
execute first 10
↓
observe again
↓
predict a new chunk
```

This is a receding-horizon policy.

### Trade-off

Small execution horizon:

- more reactive
- more inference calls
- more compute

Large execution horizon:

- cheaper
- smoother
- less responsive to new observations

---

# 6. ACT Refresher

ACT predicts an action chunk:

$$
\hat A_t =
[
\hat a_t,\dots,\hat a_{t+k-1}
]
$$

using a CVAE-based Transformer.

### What is $z$, and why set it to zero at inference?

ACT uses a **CVAE**. The latent vector $z$ captures variations in action style or timing that the observation alone cannot explain. Its source differs between training and inference.

**During training**, the demonstrated future action chunk $A$ is available. The encoder uses $A$ and current joint positions $\bar o$ (without images) to infer a posterior and sample $z$. The decoder reconstructs the chunk from the full observation $o$ and $z$:

$$
z\sim q_\phi(z\mid A,\bar o),
\qquad \hat A=f_\theta(o,z).
$$

This helps the decoder distinguish demonstration variations. The loss encourages accurate reconstruction while KL regularization keeps the posterior close to a standard Gaussian:

$$
\mathcal L=\|A-\hat A\|_1
+\beta D_{\mathrm{KL}}\big(q_\phi(z\mid A,\bar o)\,\|\,\mathcal N(0,I)\big).
$$

**During inference**, the future demonstration $A$ is unknown, so the posterior encoder cannot be used. ACT instead takes the mean of the prior $p(z)=\mathcal N(0,I)$, avoiding sampling variability and producing a deterministic output for a given observation:

$$
\boxed{z=0},\qquad \hat A=f_\theta(o,0).
$$

> Training: infer action style from demonstrations and regularize it toward the prior. Inference: use the prior mean because demonstrations are unavailable. Only the CVAE posterior encoder is removed; the policy's visual encoders remain.

### Interview follow-ups

- **Could we sample $z$?** Yes, $z\sim\mathcal N(0,I)$, but resampling at each query may change action style. ACT fixes it to zero.
- **Does zero mean the best or average action?** No. It is the latent prior's mean; nonlinear decoding need not produce the mean action, a safe action, or resolve multimodality.
- **What do chunking and temporal ensembling add?** Chunking predicts actions jointly; ensembling combines predictions for the same timestep to improve smoothness. Neither guarantees correct mode selection.
- **Why L1 rather than L2?** L1 penalizes large errors linearly, making it less sensitive to outliers than squared error. It does not guarantee jitter-free motion.

References: [ACT §IV-B–C](https://arxiv.org/html/2304.13705v1#S4.SS2), [official L1 + KL implementation](https://github.com/tonyzhaozh/act/blob/main/policy.py).

---

## 6.1 Temporal Ensembling

ACT queries the policy repeatedly.

Therefore multiple action chunks may contain predictions for the same physical timestep.

Those predictions are combined:

$$
a_t
=
\frac{
\sum_i w_i A_t^{(i)}
}{
\sum_i w_i
}
$$

with exponentially decaying weights.

### Important distinction

Temporal ensembling does **not** simply average neighboring actions.

It combines **multiple predictions of the same target timestep**.

---

# 7. Diffusion: Core Mathematics

## 7.1 Forward Noising

Let clean data be:

$$
x_0
$$

At diffusion step $k$:

$$
q(x_k\mid x_{k-1})
=
\mathcal N(
\sqrt{1-\beta_k}x_{k-1},
\beta_k I
)
$$

Define:

$$
\alpha_k=1-\beta_k
$$

and:

$$
\bar\alpha_k
=
\prod_{i=1}^{k}\alpha_i
$$

Then we can sample any noisy state directly:

$$
\boxed{
x_k
=
\sqrt{\bar\alpha_k}x_0
+
\sqrt{1-\bar\alpha_k}\epsilon
}
$$

where:

$$
\epsilon\sim\mathcal N(0,I)
$$

This is one of the equations you should memorize.

---

# 8. Why Predict Noise?

The model receives:

$$
x_k,\ k,\ c
$$

where $c$ is conditioning information.

It predicts:

$$
\epsilon_\theta(x_k,k,c)
$$

Training objective:

$$
\boxed{
\mathcal L_{\mathrm{diff}}
=
\mathbb E
[
\|
\epsilon
-
\epsilon_\theta(x_k,k,c)
\|_2^2
]
}
$$

Intuitively:

```text
clean data
 + known Gaussian noise
        ↓
      x_k
        ↓
 model learns which part was noise
        ↓
remove noise during generation
```

Noise is predicted because the added Gaussian noise is exactly known during training, provides a simple normalized regression target, and estimating it allows the clean sample to be recovered during reverse diffusion.

---

# 9. Diffusion Policy

In Diffusion Policy, the generated object is not an image.

It is an action chunk:

$$
A_t=
[
a_t,\dots,a_{t+T_p-1}
]
$$

The observation window is:

$$
O_t=
[
o_{t-T_o+1},\dots,o_t
]
$$

Noised action:

$$
A_t^k
=
\sqrt{\bar\alpha_k}A_t^0
+
\sqrt{1-\bar\alpha_k}\epsilon
$$

Network:

$$
\epsilon_\theta(
O_t,
A_t^k,
k
)
$$

Loss:

$$
\boxed{
\mathcal L
=
\mathbb E
[
\|
\epsilon
-
\epsilon_\theta(O_t,A_t^k,k)
\|_2^2
]
}
$$

---

# 10. Diffusion Policy Training Pipeline

```text
demonstration
↓
sample observation window O_t
↓
sample future clean action chunk A_t
↓
sample diffusion timestep k
↓
sample epsilon ~ N(0,I)
↓
construct noisy action A_t^k
↓
encode observation
↓
denoiser predicts epsilon_hat
↓
MSE(epsilon_hat, epsilon)
↓
backpropagation
```

Pseudocode:

```python
obs, action = batch

k = sample_timestep()
noise = torch.randn_like(action)

noisy_action = (
    sqrt_alpha_bar[k] * action
    + sqrt_one_minus_alpha_bar[k] * noise
)

noise_pred = model(
    obs=obs,
    noisy_action=noisy_action,
    timestep=k,
)

loss = mse(noise_pred, noise)
loss.backward()
optimizer.step()
```

---

# 11. Diffusion Policy Inference

```text
current observation O_t
↓
sample random Gaussian action sequence
↓
denoise step K
↓
denoise step K-1
↓
...
↓
denoise step 1
↓
final action chunk
↓
execute first H_e actions
↓
replan
```

This iterative inference is powerful but can be expensive.

That latency problem is one reason Flow Matching became attractive in newer VLA systems.

---

# 12. Flow Matching: Core Idea

Flow Matching learns a continuous vector field:

$$
v_\theta(x_t,t,c)
$$

that transports samples from one distribution to another.

Think of it as learning:

> At every position $x_t$ and time $t$, which direction should the sample move?

The dynamics are described by:

$$
\boxed{
\frac{dx_t}{dt}
=
v_\theta(x_t,t,c)
}
$$

---

# 13. Linear Interpolation Path

For robot actions, define:

- noise:

$$
x_0=\epsilon
$$

- clean action:

$$
x_1=A
$$

Use the linear path:

$$
\boxed{
x_t
=
(1-t)\epsilon+tA
}
$$

where:

$$
t\in[0,1]
$$

At:

$$
t=0
$$

we have:

$$
x_0=\epsilon
$$

At:

$$
t=1
$$

we have:

$$
x_1=A
$$

---

# 14. Deriving the Flow-Matching Target

### What does “velocity” mean?

Flow-matching velocity describes **change during generation**, not the robot's physical joint velocity $\dot q$. Generation time $t\in[0,1]$ measures progress from noise to an action sample.

For the straight interpolation path:

$$
x_t=(1-t)\epsilon+tA,
\qquad
\boxed{v^*=\frac{dx_t}{dt}=A-\epsilon}.
$$

The target is simply **endpoint minus starting point**. For example, if $A=3$ and $\epsilon=-2$, then $x_t=-2+5t$ and $v^*=5$:

```text
t:  0    0.2   0.4   0.6   0.8   1
x: -2    -1     0     1     2     3
   noise ---------------------> action
```

For an illustrative chunk $A\in\mathbb R^{50\times7}$, both noise and velocity have shape $50\times7$. The velocity tells us how to update the entire candidate chunk in generation space; its 50 rows refer to future robot steps, a separate time axis.

### Why predict this velocity?

During training, $A$ is known, so we supervise the model with:

$$
\boxed{
\mathcal L_{\mathrm{FM}}
=\mathbb E_{A,\epsilon,t,c}
\left[\|v_\theta(x_t,t,c)-(A-\epsilon)\|_2^2\right]
}.
$$

Here $c$ is conditioning information such as observations and an instruction. At inference, $A$ is unknown. Starting from $x_0\sim\mathcal N(0,I)$, the learned field supplies the direction and rate of each update:

$$
x_{t+\Delta t}\approx x_t+\Delta t\,v_\theta(x_t,t,c).
$$

This Euler step integrates an Ordinary Differential Equation (ODE) that transports noise toward the conditional action distribution. Although each training pair defines a straight path with constant target velocity, the learned field averages compatible targets at each $(x_t,t,c)$; its generated trajectories need not be straight.

### Advantages and the diffusion connection

- **Simple training:** sample a time and regress an explicit velocity target; no ODE rollout is needed during training.
- **Flexible sampling:** integrate the learned field with Euler or another ODE solver. Suitable paths can allow fewer steps, useful for latency-sensitive VLA inference.
- **Continuous chunks:** update all action dimensions jointly without discrete action tokens. Diffusion policies also support continuous outputs.

These path and solver choices are central to [Flow Matching](https://arxiv.org/abs/2210.02747). A useful intuition is **noise-predicting diffusion asks “what noise should be removed?”, while flow matching asks “which direction should the sample move?”** This is not a strict boundary: diffusion also supports other parameterizations and accelerated sampling such as [DDIM](https://arxiv.org/abs/2010.02502). Flow matching is not automatically faster or more accurate.

### Does flow matching avoid iterative generation?

**No. Both flow matching and diffusion typically generate samples over multiple steps.** Flow matching repeatedly evaluates the velocity field to integrate:

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c).
$$

A sufficiently straight, well-learned flow may require fewer integration steps at a given quality, but straight training paths do not guarantee this. Actual latency depends on the number of network evaluations, cost per forward pass (including model size and caching), solver, and implementation. Diffusion accelerations such as DDIM or distillation also affect the comparison.

> **Interview answer:** Flow matching's potential advantage is fewer steps, not zero iteration. Whether it is faster depends on the specific model and sampler at comparable output quality.

---

# 15. π0 Flow Matching

For π0:

$$
A^\tau
=
\tau A+(1-\tau)\epsilon
$$

Then:

$$
\frac{dA^\tau}{d\tau}
=
A-\epsilon
$$

The action expert predicts:

$$
v_\theta(
A^\tau,
o,
\tau
)
$$

Loss:

$$
\boxed{
\mathcal L
=
\mathbb E
[
\|
v_\theta(A^\tau,o,\tau)
-
(A-\epsilon)
\|_2^2
]
}
$$

---

# 16. Why Some Papers Write $\epsilon-A$

Suppose instead the interpolation is defined:

$$
x_t=(1-t)A+t\epsilon
$$

Then:

$$
x_0=A
$$

and:

$$
x_1=\epsilon
$$

Differentiate:

$$
\frac{dx_t}{dt}
=
\epsilon-A
$$

Therefore:

$$
v^*=\epsilon-A
$$

Nothing fundamentally changed.

The path direction changed.

---

## Interview Rule

Never memorize only the sign.

Ask:

> Which endpoint corresponds to $t=0$, and which endpoint corresponds to $t=1$?

Then differentiate the path.

---

# 17. General Interpolation Schedule

Some papers write:

$$
x_t
=
(1-\rho(t))x_0
+
\rho(t)x_1
$$

Differentiate:

$$
\frac{dx_t}{dt}
=
\rho'(t)(x_1-x_0)
$$

For:

$$
\rho(t)=t
$$

we get:

$$
\rho'(t)=1
$$

and therefore:

$$
v^*=x_1-x_0
$$

This is useful when reading papers with non-linear time schedules.

---

# 18. Flow Matching Inference

Start from Gaussian noise:

$$
x_0\sim\mathcal N(0,I)
$$

Then solve:

$$
\frac{dx_t}{dt}
=
v_\theta(x_t,t,c)
$$

The simplest solver is Euler:

$$
\boxed{
x_{t+\Delta t}
=
x_t
+
\Delta t
v_\theta(x_t,t,c)
}
$$

Repeatedly:

```text
noise
↓
Euler step
↓
slightly more action-like
↓
Euler step
↓
...
↓
clean action chunk
```

---

# 19. Minimal Flow Matching Training Code

You should be able to understand this immediately:

```python
def flow_matching_loss(model, obs, action):
    batch_size = action.shape[0]

    t = torch.rand(
        batch_size,
        1,
        1,
        device=action.device,
    )

    noise = torch.randn_like(action)

    noisy_action = (
        (1 - t) * noise
        + t * action
    )

    target_velocity = action - noise

    pred_velocity = model(
        obs=obs,
        noisy_action=noisy_action,
        timestep=t,
    )

    loss = (
        pred_velocity
        - target_velocity
    ).pow(2).mean()

    return loss
```

---

# 20. Minimal Flow Matching Inference Code

```python
@torch.no_grad()
def sample_action(model, obs, shape, steps=10):
    x = torch.randn(shape, device=obs.device)

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full(
            (shape[0], 1, 1),
            i / steps,
            device=obs.device,
        )

        velocity = model(
            obs=obs,
            noisy_action=x,
            timestep=t,
        )

        x = x + dt * velocity

    return x
```

This is conceptually close to what π0-style action generation is doing, although real implementations contain:

- transformer attention
- action/state embeddings
- masks
- conditioning
- normalization
- efficient caching
- better numerical details

---

# 21. Diffusion vs Flow Matching

| Aspect | Diffusion | Flow Matching |
|---|---|---|
| Start distribution | Gaussian noise | Gaussian noise |
| End distribution | Data/action | Data/action |
| Training object | noisy samples | interpolated samples |
| Common target | noise $\epsilon$ | velocity $dx/dt$ |
| Typical dynamics | stochastic/diffusion process | ODE/vector field |
| Loss | usually MSE | usually MSE |
| Conditioning | supported | supported |
| Multimodal generation | yes | yes |
| Iterative inference | yes | yes |
| Modern VLA example | Diffusion Policy | π0 |
| Main mental model | remove noise | move sample along learned flow |

---

# 22. Important Similarity

Do not overstate the difference.

Both methods:

1. begin from a simple noise distribution;
2. construct intermediate noisy/interpolated samples;
3. condition on observations;
4. train a neural network with a regression objective;
5. iteratively transform noise into an action sample.

Therefore Flow Matching is not "completely unrelated" to diffusion.

The difference is mainly the mathematical formulation of the transport dynamics and the prediction target.

---

# 23. Why Generative Policies Help Robot Actions

Robot demonstrations often contain:

$$
p(A\mid O)
$$

that is multimodal.

A deterministic regression model approximates:

$$
\mathbb E[A\mid O]
$$

which may average incompatible behaviors.

Generative models instead attempt to model the full distribution:

$$
A\sim p_\theta(A\mid O)
$$

Therefore different random initializations can produce different valid trajectories.

---

# 24. Continuous vs Discrete Action Modeling

Modern VLA systems mainly use two families.

## Continuous

Examples:

- Diffusion Policy
- π0
- some WALL variants

Output:

$$
A\in\mathbb R^{H\times d_a}
$$

Advantages:

- preserves numerical precision
- natural for robot control
- good for high-frequency trajectories

Disadvantages:

- requires specialized action head
- often iterative inference
- integration with LLM/VLM token prediction is less direct

---

## Discrete / Autoregressive

Examples:

- RT-2
- OpenVLA
- FAST-based models

Convert actions into tokens:

$$
A
\rightarrow
z_1,z_2,\dots,z_K
$$

and optimize:

$$
\mathcal L_{\mathrm{CE}}
=
-\sum_k
\log
p_\theta(
z_k
\mid
z_{<k},o
)
$$

Advantages:

- directly reuses VLM next-token prediction
- unified language/action interface
- easy to combine with text outputs

Disadvantages:

- quantization error
- autoregressive decoding latency
- naive tokenization may create long token sequences
- continuous precision becomes harder

---

# 25. RT-2 / OpenVLA-Style Action Tokenization

A simple approach:

1. take each continuous action dimension;
2. clip to a range;
3. divide into bins;
4. map each bin to a vocabulary token.

Suppose one dimension is:

$$
a\in[a_{\min},a_{\max}]
$$

with $B$ bins.

A simple bin index is approximately:

$$
b
=
\left\lfloor
B
\frac{
a-a_{\min}
}{
a_{\max}-a_{\min}
}
\right\rfloor
$$

Then:

$$
b
\rightarrow
\text{token}
$$

For a 7D action:

```text
dx dy dz droll dpitch dyaw gripper
```

one control step may become seven tokens.

---

# 26. Why Naive Action Tokens Are Expensive

Suppose:

- 7 action dimensions
- 50-step action chunk

Then naive per-dimension tokenization needs about:

$$
7\times 50=350
$$

action tokens.

Autoregressive decoding of 350 tokens can be expensive.

This motivated more efficient representations such as FAST.

---

# 27. FAST Action Tokenization

FAST approximately follows:

```text
continuous action trajectory
↓
normalization
↓
DCT
↓
frequency coefficients
↓
quantization
↓
integer symbols
↓
BPE
↓
action tokens
```

---

# 28. Why DCT?

Robot trajectories are temporally smooth.

Neighboring timesteps contain strong redundancy.

DCT transforms the trajectory from the time domain into frequency coefficients.

Instead of directly storing:

$$
[a_1,a_2,\dots,a_H]
$$

we represent it using frequency information.

Low-frequency components often capture most of the trajectory structure.

---

# 29. Why BPE?

After quantization, frequently occurring sequences of action symbols can be merged into larger tokens.

Therefore:

```text
many primitive action symbols
↓
repeated patterns
↓
BPE merges
↓
shorter token sequence
```

This reduces autoregressive sequence length.

---

# 30. FAST vs Flow Matching

FAST:

$$
A
\rightarrow
\text{tokens}
$$

then:

$$
p(z_k\mid z_{<k},o)
$$

Flow matching:

$$
\epsilon
\rightarrow
A
$$

through:

$$
\frac{dx_t}{dt}
=
v_\theta(x_t,t,o)
$$

FAST is fundamentally a **discrete sequence modeling** approach.

Flow Matching is fundamentally a **continuous generative modeling** approach.

---

# 31. Action Normalization

Action normalization matters for both diffusion and flow models.

Suppose joint/action dimensions have very different scales:

```text
translation: 0.01 m
rotation:    0.3 rad
gripper:     1.0
```

If used directly, loss magnitude can be dominated by certain dimensions.

Common transformations include:

## Standardization

$$
\tilde a_i
=
\frac{
a_i-\mu_i
}{
\sigma_i+\epsilon
}
$$

## Min-max scaling

$$
\tilde a_i
=
2
\frac{
a_i-a_i^{\min}
}{
a_i^{\max}-a_i^{\min}
}
-1
$$

## Quantile normalization

Use robust training-set percentiles instead of raw minima/maxima.

The inverse transformation must be applied before sending actions to the robot.

---

# 32. One Training Batch: What Is Actually Inside?

A useful mental model:

```python
batch = {
    "images": ...,
    "language": ...,
    "state": ...,
    "actions": ...,
    "action_mask": ...,
}
```

Example shapes:

```text
images:
[B, num_cameras, C, H, W]

state:
[B, state_dim]

actions:
[B, action_horizon, action_dim]

action_mask:
[B, action_horizon, action_dim]
```

For a flow policy we additionally sample:

```text
t:
[B, 1, 1]

noise:
[B, action_horizon, action_dim]
```

Then construct:

```text
noisy_action:
[B, action_horizon, action_dim]
```

The network finally predicts:

```text
velocity:
[B, action_horizon, action_dim]
```

---

# 33. π0 Training Step — Conceptual View

```text
images
language
robot state
clean action chunk
        ↓
sample flow time t
        ↓
sample Gaussian noise
        ↓
interpolate noisy action
        ↓
VLM encodes images/language
        ↓
action expert receives:
    state
    noisy actions
    flow time
    VLM context
        ↓
predict velocity
        ↓
MSE against target velocity
        ↓
backpropagation
```

---

# 34. π0 Inference Step — Conceptual View

```text
current images
language
robot state
        ↓
encode observation
        ↓
initialize random action noise
        ↓
flow step 1
        ↓
flow step 2
        ↓
...
        ↓
continuous action chunk
        ↓
denormalize
        ↓
send actions to controller
```

---

# 35. Interview Comparison: Four Action Policy Families

| Policy family | Example | Output | Loss | Multimodal? | Typical weakness |
|---|---|---|---|---|---|
| Regression | simple BC | continuous action | L1/MSE | weak | mode averaging |
| CVAE | ACT | continuous action chunk | reconstruction + KL | yes | latent modeling complexity |
| Diffusion | Diffusion Policy | continuous chunk | noise MSE | strong | iterative inference |
| Flow Matching | π0 | continuous chunk | velocity MSE | strong | iterative ODE sampling |
| Autoregressive tokens | RT-2/OpenVLA | tokens | CE | yes | quantization + AR latency |
| FAST tokens | FAST-based VLA | compressed action tokens | CE | yes | still discrete / AR |

---

# 36. Frequently Confused Concepts

## Diffusion timestep vs robot timestep

Diffusion step $k$:

> denoising stage inside the generative process

Robot timestep $t$:

> physical time/control step

They are unrelated variables.

---

## Flow time vs robot time

Flow variable $\tau$:

> interpolation from noise to action

Robot time $t$:

> actual trajectory timestep

Do not confuse them.

---

## Action horizon vs number of flow steps

Action horizon:

$$
H
$$

means:

> how many future robot actions are generated.

Flow steps:

$$
N
$$

means:

> how many numerical integration steps are used to generate the chunk.

Example:

```text
action horizon = 50
flow steps = 10
```

does **not** mean ten of the fifty robot actions are generated.

Each flow step updates the entire 50-step action chunk.

---

# 37. Very Important Interview Question

## "π0 predicts 50 actions. If it uses 10 flow steps, what exactly happens?"

Correct answer:

The model initializes the **whole action chunk**:

$$
A^0
\in
\mathbb R^{50\times d_a}
$$

as Gaussian noise.

Each of the 10 flow iterations predicts a velocity for the entire tensor:

$$
v_\theta
\in
\mathbb R^{50\times d_a}
$$

and updates all 50 actions simultaneously.

After 10 iterations, the entire chunk becomes a clean action trajectory.

The 10 flow steps are generative inference iterations, not physical action timesteps.

---

# 38. Why Flow Matching Fits VLA

VLA must turn visual and language understanding into coordinated robot actions. Flow matching fits several requirements of this output space:

1. **Continuous actions need precision.** Joint positions and end-effector displacements are continuous. Generating $A\in\mathbb R^{H\times D}$ directly, where $H$ is the horizon and $D$ the action dimension, avoids the quantization error introduced by discretizing actions into token bins. Prediction errors can still remain.

2. **One task can admit several valid trajectories.** A conditional flow can transform different noise samples into different action chunks, modeling $p(A\mid O,L)$ for observations $O$ and language $L$. Unlike direct MSE regression, it need not reduce these alternatives to one conditional mean. Random initialization alone does not guarantee that all modes are learned.

3. **Actions must be coordinated across time and dimensions.** Treating the entire $H\times D$ chunk as one sample lets a sequence model learn dependencies between future steps, joints, and gripper commands.

4. **The generator needs task and scene context.** VLM features can condition the velocity field, alongside robot state where needed:

   $$
   c=f_{\mathrm{VLM}}(O,L),\qquad v_\theta(x_t,t,c).
   $$

   This connects pretrained semantic representations to continuous action generation, as in [π0](https://arxiv.org/abs/2410.24164).

5. **Long chunks make sequential token decoding expensive.** Each flow evaluation updates the entire candidate chunk in parallel:

   $$
   x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t,c),
   \qquad x_t\in\mathbb R^{H\times D}.
   $$

   This avoids an autoregressive dependency between successive action tokens, though generation still requires integration steps. Actual speed depends on the model and sampler, as discussed in Section 14.

### Why flow matching specifically?

Continuous outputs, multimodality, and joint chunk generation are also strengths of [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). Flow matching offers a direct velocity-regression objective and flexible ODE paths and solvers; suitable flows may need fewer sampling steps, but superiority over diffusion is not guaranteed.

> **Interview answer:** Continuous generative policies fit VLA because they can produce coordinated, multimodal action chunks under semantic conditioning. Flow matching is one suitable implementation: it avoids action-bin discretization and updates the whole chunk through a learned velocity field.

---

# 39. Why Not Just Use an MLP?

A direct MLP could predict:

$$
\hat A=f_\theta(o)
$$

but this often implies a relatively simple conditional action distribution.

Generative policies provide a richer:

$$
p(A\mid o)
$$

and can represent multiple valid trajectories.

That is especially important in:

- contact-rich tasks
- clutter
- deformable objects
- bimanual manipulation
- demonstrations with multiple strategies

---

# 40. Week 1 Coding Exercise

Create:

```text
src/interview/flow_matching_toy.py
```

Recommended toy problem:

Generate 2D samples from a multimodal target distribution.

Example targets:

```text
cluster 1: (-2, 0)
cluster 2: (+2, 0)
```

Training:

```text
target point x1
noise x0 ~ N(0,I)
t ~ Uniform(0,1)

xt = (1-t)x0 + t x1

target velocity:
v = x1 - x0
```

Train a small MLP:

```text
input:
xt
t

output:
vx, vy
```

Inference:

```text
sample noise
integrate Euler
plot generated points
```

### Why this exercise matters

If you can implement this toy model yourself, then the π0 Flow Matching formula stops being something you only memorized.

---

# 41. Whiteboard Derivation Drill

Without notes, write:

### Flow path

$$
x_t=(1-t)\epsilon+tx_1
$$

### Differentiate

$$
\frac{dx_t}{dt}
=
x_1-\epsilon
$$

### Objective

$$
\mathcal L=
\mathbb E
[
\|
v_\theta(x_t,t,c)
-
(x_1-\epsilon)
\|^2
]
$$

### Inference

$$
x_{t+\Delta t}
=
x_t+
\Delta t
v_\theta(x_t,t,c)
$$

You should be able to finish this in **under 90 seconds**.

---

# 42. Interview Questions — Basic

These reference answers are written for interviews: accurate and concise, with enough detail to handle follow-up questions.

### Q1. What is Behavior Cloning?

Behavior Cloning (BC) turns imitation learning into supervised learning. Given expert data $(o,a)$, we train a policy to learn:

$$
\pi_\theta(a\mid o).
$$

The basic objective is to maximize the likelihood of expert actions:

$$
\max_\theta\mathbb E_{(o,a)\sim D}
[\log\pi_\theta(a\mid o)].
$$

If actions follow a Gaussian distribution with fixed variance, maximum likelihood is equivalent to minimizing the MSE between predicted and expert actions.

---

### Q2. Why does one-step MSE behavior cloning struggle with multimodal actions?

MSE tends to predict the conditional mean:

$$
\hat a\approx\mathbb E[a\mid o].
$$

However, the same observation may admit several reasonable actions. For example, the robot can go either left or right around an obstacle:

$$
a_1=-1,\qquad a_2=+1.
$$

MSE may predict:

$$
\hat a=0,
$$

which could send the robot straight into the obstacle.

One advantage of generative models such as diffusion, flow matching, and CVAEs is their ability to learn the full distribution:

$$
p(a\mid o),
$$

rather than predicting only a mean.

---

### Q3. What problem does action chunking solve?

Instead of predicting only the next action $a_t$, action chunking predicts a sequence of future actions:

$$
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}].
$$

It has three main benefits:

1. It exploits temporal correlations between actions, making trajectories more coherent.
2. It reduces the effective decision horizon, avoiding an independent decision at every control step.
3. It can learn complete short motion patterns such as reaching, grasping, and lifting.

---

### Q4. Does action chunking completely solve covariate shift?

No. Action chunking can reduce error accumulation from frequent independent decisions, but the policy is still trained mainly on the state distribution of expert demonstrations:

$$
d_{\pi_E}(s).
$$

At deployment, the robot visits states induced by its own policy:

$$
d_{\pi_\theta}(s).
$$

If execution errors take the robot into states not covered by the training data, distribution shift still occurs.

---

### Q5. What is the difference between prediction horizon and execution horizon?

Prediction horizon $H_p$ is:

> How many future actions the model predicts in one call.

Execution horizon $H_e$ is:

> How many of those actions are actually executed before observing and running inference again.

For example:

$$
H_p=50,\qquad H_e=10.
$$

```text
Observation
→ predict 50 actions
→ execute first 10
→ new observation
→ predict another 50
```

Therefore, the system can remain closed-loop even when it predicts a long action chunk.

---

### Q6. Why does Diffusion Policy generate an action sequence instead of a single action?

Robot actions have strong temporal correlations. Generating a whole action chunk:

$$
A_t=[a_t,\ldots,a_{t+H-1}]
$$

jointly models relationships between future actions, making trajectories more coherent while reducing the effective decision horizon.

Diffusion can also model a multimodal action distribution:

$$
p(A_t\mid O_t),
$$

rather than predicting only one deterministic action.

---

### Q7. Write the DDPM forward-noising equation.

The key equation to remember is:

$$
\boxed{
x_k=\sqrt{\bar\alpha_k}x_0+
\sqrt{1-\bar\alpha_k}\epsilon
}
$$

where:

$$
\epsilon\sim\mathcal N(0,I),
$$

and:

$$
\alpha_k=1-\beta_k,\qquad
\bar\alpha_k=\prod_{i=1}^{k}\alpha_i.
$$

For Diffusion Policy, $x_0$ can be understood as a clean action chunk.

---

### Q8. What does the diffusion model predict?

The most common formulation of classical DDPM predicts the added Gaussian noise:

$$
\epsilon_\theta(x_k,k,c).
$$

The training loss is:

$$
\mathcal L=\mathbb E
\left[\|\epsilon-\epsilon_\theta(x_k,k,c)\|^2\right],
$$

where $c$ contains conditioning information such as observations.

However, **diffusion is not limited to noise prediction**. Other parameterizations predict $x_0$, velocity, and related quantities.

In an interview, say that a common classical DDPM / Diffusion Policy formulation predicts noise, rather than claiming that all diffusion models must predict noise.

---

### Q9. What does Flow Matching predict?

Flow matching directly learns a velocity field:

$$
v_\theta(x_t,t,c),
$$

which describes:

> In which direction, and at what rate, the current sample $x_t$ should move at generation time $t$.

It defines the ODE:

$$
\boxed{\frac{dx_t}{dt}=v_\theta(x_t,t,c)}.
$$

At inference, integrating this velocity field transports noise into an action sample.

---

### Q10. What is the π0 Flow Matching target?

If we define:

$$
A^\tau=(1-\tau)\epsilon+\tau A,
$$

then:

$$
\frac{dA^\tau}{d\tau}=A-\epsilon.
$$

Therefore, the target velocity is:

$$
\boxed{v^*=A-\epsilon}.
$$

The loss is:

$$
\boxed{
\mathcal L=\mathbb E
\left[\|v_\theta(A^\tau,o,\tau)-(A-\epsilon)\|^2\right]
}.
$$

---

# 43. Interview Questions — Intermediate

### Q11. Derive the Flow Matching target rather than quoting it.

This is one of the most important questions to master in Week 1.

First, define a linear path from noise to data/action:

$$
x_t=(1-t)\epsilon+tx_1.
$$

Differentiate with respect to $t$:

$$
\frac{dx_t}{dt}=-\epsilon+x_1=x_1-\epsilon.
$$

Therefore, the target velocity is:

$$
\boxed{v^*=x_1-\epsilon}.
$$

The model learns:

$$
v_\theta(x_t,t,c)\approx x_1-\epsilon,
$$

with the loss:

$$
\mathcal L=\mathbb E
\left[\|v_\theta(x_t,t,c)-(x_1-\epsilon)\|^2\right].
$$

For VLA, $x_1=A$, so:

$$
v^*=A-\epsilon.
$$

**Do not memorize the formula alone. If the interviewer changes the notation, differentiate the interpolation path again.**

---

### Q12. Why can one paper use $A-\epsilon$ and another use $\epsilon-A$?

They may define opposite flow-time directions.

If:

$$
x_t=(1-t)\epsilon+tA,
$$

then:

$$
\frac{dx_t}{dt}=A-\epsilon.
$$

But if:

$$
x_t=(1-t)A+t\epsilon,
$$

then:

$$
\frac{dx_t}{dt}=\epsilon-A.
$$

Do not judge correctness from the sign alone. First check whether $t=0$ and $t=1$ correspond to noise or data, then differentiate the interpolation path.

---

### Q13. What is the difference between a diffusion step and a robot control step?

They refer to entirely different time scales.

A diffusion step is:

> One internal denoising iteration used to generate actions from noise.

A robot control step is:

> One action execution step in physical time.

For example:

```text
10 diffusion steps
→ generate 50 robot actions
→ execute 10 robot control steps
```

The numbers 10, 50, and 10 describe different concepts.

---

### Q14. What is the difference between the number of flow steps and action horizon?

Action horizon $H$ is:

> The number of future robot actions in one action chunk.

Flow steps $N$ is:

> The number of ODE integration updates used to generate that chunk from noise.

For example:

$$
H=50,\qquad N=10.
$$

We initialize the entire chunk:

$$
A^0\in\mathbb R^{50\times d_a}.
$$

Every flow step updates **the entire 50-step action chunk**:

$$
A^{k+1}=A^k+\Delta t\,v_\theta(A^k,t,o).
$$

It does not mean that one flow step generates one robot action. This is a common follow-up question.

---

### Q15. Why does action normalization matter for generative policies?

Different action dimensions can have very different numerical ranges. For example:

$$
\Delta x\approx0.01,\qquad
\Delta\theta\approx0.5,
$$

while the gripper value may range from 0 to 1.

Without normalization, larger-scale dimensions dominate MSE, and the model has more difficulty learning noise/velocity distributions on a consistent scale.

We therefore typically transform:

$$
a\rightarrow\tilde a,
$$

perform training and generation in normalized action space, and then invert the transform before deployment:

$$
\tilde a\rightarrow a.
$$

The resulting actions are sent to the robot controller.

---

### Q16. Why does a generative action policy handle multimodality better than MSE regression?

Deterministic MSE regression tends to learn the average behavior:

$$
\mathbb E[A\mid O].
$$

A generative policy learns:

$$
p(A\mid O).
$$

For the same observation, this distribution may contain both a “grasp from the left” mode and a “grasp from the right” mode.

Diffusion and flow matching can start from different noise samples and produce different but reasonable trajectories, without averaging them into a potentially invalid action.

---

### Q17. What are the major advantages of discrete action tokens?

The main advantage is:

> **They turn robot control into next-token prediction, a task that pretrained VLMs are already well suited for.**

Continuous actions:

$$
A\in\mathbb R^d
$$

are converted into:

$$
z_1,z_2,\ldots,z_K.
$$

We can then model:

$$
p(z_k\mid z_{<k},O,L)
$$

and train with cross-entropy.

This allows extensive reuse of pretrained LLM/VLM architectures:

```text
image
language
 action tokens
       ↓
same Transformer
       ↓
next-token prediction
```

This is an important attraction of the RT-2 / OpenVLA approach.

---

### Q18. What information is lost by quantizing continuous actions?

Precision.

An original action might be:

$$
a=0.1374.
$$

After quantization, it is assigned to a bin:

$$
a\rightarrow b_{37}.
$$

Decoding that bin might produce:

$$
\hat a=0.14,
$$

so:

$$
a\ne\hat a.
$$

This quantization error can be particularly important in fine robotic manipulation, where numerical action precision matters more directly than in ordinary language tasks.

---

### Q19. Why can autoregressive action generation be slow?

Tokens must be decoded sequentially:

$$
z_1\rightarrow z_2\rightarrow z_3\rightarrow\cdots.
$$

Each later token depends on earlier tokens:

$$
p(z_k\mid z_{<k},O).
$$

Therefore, it cannot output the whole action chunk in parallel like a standard continuous prediction head.

For example:

$$
50\text{ steps}\times7\text{ dimensions}=350
$$

action tokens can require many autoregressive decoding steps. This is one of the problems FAST addresses.

---

### Q20. Why does FAST use DCT?

Robot action trajectories are typically temporally smooth, with strong correlations between neighboring timesteps. Directly tokenizing:

$$
[a_1,a_2,\ldots,a_H]
$$

leaves substantial temporal redundancy.

DCT transforms the trajectory from the time domain into the frequency domain:

$$
A\xrightarrow{\mathrm{DCT}}C.
$$

Information in a smooth trajectory often concentrates in lower-frequency coefficients. The pipeline:

$$
\text{DCT}\rightarrow\text{quantization}\rightarrow\text{BPE}
$$

makes action sequences easier to compress into fewer tokens, reducing the length and cost of autoregressive action decoding.

---

# 44. Interview Questions — Advanced Follow-Ups

These questions test more than formula recall:

> **Have you actually trained and debugged models?**

---

### Q21. Suppose your Flow Matching policy becomes unstable during inference. What would you inspect?

I would investigate systematically through **Data → Training → Sampling → Deployment**, rather than guessing.

First, check action preprocessing: training and inference must use consistent normalization/denormalization, coordinate frames, units, and dimension ordering.

Then check the flow-matching formulation. If training defines:

$$
x_t=(1-t)\epsilon+tA
$$

with target:

$$
A-\epsilon,
$$

inference must integrate in the noise-to-action direction. Reversing the timestep convention or velocity sign will break sampling.

Next, inspect numerical integration: is $\Delta t$ too large, are there too few flow steps, or is the solver unstable?

Finally, inspect the model and data:

- Is the timestep embedding correct?
- Are padding and action masks correct?
- Does the dataset contain abnormal actions?
- Are observations and actions synchronized?
- Are deployment observations outside the training distribution?

**One-sentence interview answer:**

> I would first verify preprocessing and coordinate conventions, then check the flow direction and timestep convention, then inspect numerical integration, and finally determine whether the instability comes from the model or an out-of-distribution deployment state.

---

### Q22. Your policy loss decreases but real-robot success does not improve. Why?

This is an important question because:

$$
\text{training loss}\ne\text{task success}.
$$

A lower flow-matching MSE only means that $v_\theta$ is closer to target velocities on the training data. It does not guarantee successful closed-loop rollouts.

I would investigate four levels.

**1. Data**

The training data may have:

- insufficient coverage;
- poor demonstrations;
- too few recovery states;
- train/test leakage;
- task imbalance.

**2. Model**

Possible issues include:

- overfitting;
- an unsuitable action horizon;
- conditioning that is not actually being used;
- inappropriate loss weighting.

**3. Offline-to-online gap**

Offline action error may be small, but a small execution error can cause:

$$
s_t\rightarrow s_{t+1}^{\mathrm{OOD}},
$$

where OOD means out of distribution. Errors may then accumulate.

**4. Deployment**

Real robots introduce additional factors:

- inference latency;
- camera latency;
- action delay;
- calibration error;
- coordinate-frame mismatch;
- controller dynamics;
- sensor noise.

We should therefore monitor all of the following, rather than loss alone:

$$
\text{offline loss}
+\text{offline metrics}
+\text{real rollout success}
+\text{failure taxonomy}.
$$

---

### Q23. Why can predicting a 50-step chunk still be closed-loop control?

Because:

> **Predicting 50 actions does not mean executing all 50.**

For example:

$$
H_p=50,\qquad H_e=10.
$$

The system operates as follows:

```text
O_t
 ↓
predict [a_t ... a_{t+49}]
 ↓
execute [a_t ... a_{t+9}]
 ↓
O_{t+10}
 ↓
predict again
```

New observations continually provide feedback:

$$
O\rightarrow\pi\rightarrow A\rightarrow\text{robot}\rightarrow O'.
$$

This is still closed-loop control. Executing all 50 actions without observing again during the chunk would instead be open-loop chunk execution.

---

### Q24. Why would π0 use continuous Flow Matching instead of OpenVLA-style tokens?

This is a useful comparison to prepare for interviews.

OpenVLA-style methods first convert:

$$
A_{\mathrm{continuous}}\rightarrow A_{\mathrm{tokens}},
$$

then use the VLM's next-token prediction. The advantage is strong reuse of the pretrained VLM architecture.

However, this introduces quantization error, and long action chunks may require many autoregressive tokens.

In π0, the pretrained VLM provides:

> Visual and language semantic understanding.

An action expert then uses flow matching to generate:

$$
A\in\mathbb R^{H\times d_a}.
$$

This models the whole chunk directly in continuous action space, avoids the precision loss of simple discretization, and is well suited to high-frequency, continuous, coordinated robot motion.

The trade-off is:

> An additional continuous action expert and multiple flow-matching integration steps during inference.

It is therefore not a claim that flow matching is always better than action tokens. They represent different trade-offs:

$$
\boxed{\text{AR tokens: VLM-native}}
$$

versus:

$$
\boxed{\text{Flow Matching: continuous-control-native}}.
$$

---

### Which of these 24 questions should you prioritize?

If an interview is coming up soon, focus on **Q7, Q10–Q16, and Q20–Q24**. In particular, practice deriving the flow-matching target in Q11 on a whiteboard in **30–60 seconds**, rather than memorizing $A-\epsilon$.

For Q21 and Q22, connect your answers to your own RM65 project. Evidence that you have worked through **training → real-robot evaluation → failure analysis** can lead naturally to deeper questions about your practical experience.

---

# 45. Map This Week to Your RM65 Project

Your own project is a very good place to anchor these concepts.

You can explain your policy as roughly:

```text
images + instruction + robot state
↓
VLM / multimodal backbone
↓
Plan prediction
↓
continuous action expert
↓
Flow Matching action chunk
```

Important questions to prepare:

1. What is the exact action representation?
2. Absolute joint target, delta joint, or EE delta?
3. What is action dimension?
4. What is action horizon?
5. How is action normalized?
6. How are actions truncated at subtask boundaries?
7. What does the Flow Matching loss look like?
8. What is the noise distribution?
9. How is flow time sampled?
10. How many inference steps?
11. How many actions are executed before replanning?
12. How does Plan conditioning affect the action expert?
13. Are Plan tokens visible to action tokens through attention?
14. What happens if Plan prediction is wrong?
15. Why are Box/FAST auxiliary heads removed at inference?

Mark any question you currently cannot answer immediately.

---

# 46. Notes You Should Add to Your Repository

Recommended new note:

```text
docs/VLA_Interview/01_Action_Modeling_Foundations.md
```

Sections:

```text
1. Behavior Cloning
2. Action Chunking
3. Diffusion
4. Diffusion Policy
5. Flow Matching
6. π0
7. Action Tokenization
8. FAST
9. Comparison
10. Interview Questions
```

This Week 1 file can become the first version.

---

# 47. Weekly Self-Test

Score each item:

- `0`: cannot answer
- `1`: understand after seeing notes
- `2`: can explain without notes
- `3`: can derive / implement / handle follow-ups

| Topic | Score |
|---|---:|
| BC objective | |
| Covariate shift | |
| Multimodal actions | |
| Action chunking | |
| ACT temporal ensemble | |
| DDPM forward equation | |
| Diffusion loss | |
| Diffusion Policy pipeline | |
| Flow interpolation path | |
| Flow target derivation | |
| Euler integration | |
| π0 Flow Matching | |
| Sign/time convention | |
| Action normalization | |
| AR action tokenization | |
| RT-2/OpenVLA tokens | |
| FAST | |
| Diffusion vs Flow | |
| Continuous vs discrete | |
| RM65 project mapping | |

Target:

$$
\text{average score} \ge 2.3
$$

before moving on.

---

# 48. Week 1 Final Checklist

You are ready for Week 2 when you can do all of these:

- [ ] derive BC from maximum likelihood
- [ ] explain covariate shift
- [ ] explain why MSE averages modes
- [ ] explain action chunking
- [ ] distinguish prediction and execution horizons
- [ ] write DDPM forward equation
- [ ] write diffusion noise-prediction loss
- [ ] explain Diffusion Policy training
- [ ] derive Flow Matching velocity
- [ ] explain $A-\epsilon$ vs $\epsilon-A$
- [ ] write Euler inference
- [ ] explain π0's action expert
- [ ] distinguish flow steps from action timesteps
- [ ] explain action normalization
- [ ] explain RT-2/OpenVLA action tokens
- [ ] explain FAST's DCT + quantization + BPE
- [ ] compare regression / diffusion / flow / AR
- [ ] write a toy Flow Matching model
- [ ] explain your RM65 action training pipeline
- [ ] answer the 24 interview questions orally

---

# 49. One-Minute Summary

If you only review one block before an interview, use this:

```text
Behavior Cloning:
learn p(a|o) from demonstrations.
Simple MSE assumes a unimodal Gaussian and can average multiple valid actions.

Action Chunking:
predict a future action sequence instead of one action.
It improves temporal consistency and reduces effective decision horizon.

Diffusion Policy:
add Gaussian noise to clean action chunks and train a conditional model
to predict the added noise.
Inference starts from random actions and repeatedly denoises.

Flow Matching / π0:
interpolate noise and clean actions:

    x_t = (1-t) epsilon + t A

therefore:

    dx_t/dt = A - epsilon

Train the model to predict this velocity.
Inference integrates the learned vector field from noise to action.

Autoregressive VLA:
continuous actions are quantized into tokens and trained using next-token CE.
RT-2/OpenVLA use discrete action tokens.

FAST:
compresses smooth action trajectories using normalization + DCT +
quantization + BPE to reduce token length.

Main trade-off:
continuous generative models preserve precision;
discrete token models integrate naturally with pretrained VLMs.
```

---

# 50. Suggested Next Step

After finishing this note, Week 2 should focus on:

**VLA Architecture Evolution**

```text
ACT
→ Diffusion Policy
→ RT-1 / RT-2
→ Octo
→ OpenVLA
→ π0
→ FAST
→ π0.5
→ SmolVLA
→ RDT
→ GR00T
→ WALL-OSS
```

For every model, answer only six questions first:

1. Input?
2. Backbone?
3. Action representation?
4. Training loss?
5. Training data?
6. Inference procedure?

That structure prevents paper details from becoming disconnected facts.

## Model Comparison: The Six Questions

Read each row from left to right: inputs determine the required representation, the backbone connects them to an action space, and the loss/data explain how the inference procedure is learned. **CE** = cross-entropy; **FM** = flow matching; **EE** = end effector; **OXE** = Open X-Embodiment. For diffusion/FM models, the action generator additionally receives the current noisy chunk and its noise/flow time.

**Scope:** ACT and the original Diffusion Policy are visuomotor precursors without language conditioning. FAST is an action tokenizer, so its row uses **π0-FAST** as the policy example. RDT means **RDT-1B**; GR00T means **GR00T N1**, following the repository note's v2 paper. Other rows describe the original named methods, not later variants such as OpenVLA-OFT. Listed horizons and sampling steps are reported configurations, not universal requirements.

| Model / method | 1. Input? | 2. Backbone? | 3. Action representation? | 4. Training loss? | 5. Training data? | 6. Inference procedure? |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **[ACT](../docs/Model_Zoo/Robotics/Policies/ACT.md)** · [paper](https://arxiv.org/abs/2304.13705) | Four RGB views + current joint positions; no language. The training-only CVAE encoder also sees the demonstrated chunk. | ResNet-18 vision encoders + Transformer encoder/decoder policy; separate CVAE posterior encoder. | Continuous target joint-position chunks, including grippers; ALOHA: 14 dimensions, typically 100 actions. | L1 action reconstruction + weighted KL to a standard Gaussian prior. | Task-specific ALOHA teleoperation: typically 50 demonstrations per real task, 100 for Thread Velcro; separate simulation demonstrations. | Discard the posterior encoder, set $z=0$, predict a chunk; repeatedly query and temporally ensemble predictions for the same execution timestep. |
| **[Diffusion Policy](../docs/Model_Zoo/Robotics/Policies/Diffusion_Policy.md)** · [paper](https://arxiv.org/abs/2303.04137) | Recent RGB observations + robot state, or low-dimensional observations; no language in the original method. | ResNet-18 for visual observations + conditional temporal 1D U-Net or diffusion Transformer. | Continuous action chunks; position/velocity and joint/EE representation depend on the task. | Conditional denoising MSE, commonly predicting added Gaussian noise. | Expert demonstrations for individual benchmarks: Robomimic, Push-T, Block Pushing, Franka Kitchen, and real manipulation tasks; no shared web-scale VLA pretraining recipe. | Encode observations once, iteratively denoise an entire chunk from Gaussian noise; execute a prefix and replan with fresh observations. |
| **[RT-1](../docs/Model_Zoo/Robotics/Policies/RT_1.md)** · [paper](https://arxiv.org/abs/2212.06817) | Six-frame RGB history + language instruction. | USE language embedding → FiLM-conditioned EfficientNet-B3 → TokenLearner → 8-layer Transformer. | One control step: 7 arm/gripper dimensions, 3 base dimensions, and a mode/termination variable; continuous dimensions use 256 bins. | Per-dimension categorical CE on demonstrated actions. | About 130k demonstrations, 13 robots, 744 instructions, collected over 17 months. | Predict action dimensions without autoregressive action-token conditioning, decode bins, execute one step, and refresh the image history; about 3 Hz control. |
| **[RT-2](../docs/Model_Zoo/Robotics/Policies/RT_2.md)** · [paper](https://arxiv.org/abs/2307.15818) | Current RGB image(s) + language instruction. | Pretrained PaLI-X or PaLM-E VLM; reuses its token-generation interface. | One action encoded as discrete tokens: termination, EE translation/rotation, and gripper; 256-bin continuous dimensions. | Next-token CE on robot action strings and web vision-language outputs during co-fine-tuning. | RT-1 robot trajectories mixed with the original VLM's web vision-language data, following VLM pretraining. | Autoregressively decode valid action tokens, convert to controls, execute, and observe again. |
| **[Octo](../docs/Model_Zoo/Robotics/Policies/Octo.md)** · [paper](https://arxiv.org/abs/2405.12213) | Short RGB history, typically two frames, with available camera masks; task specified by language or a goal image. | Image patch tokenizers + frozen T5-base language encoder + block-masked Transformer with readout tokens + small MLP diffusion head. | Continuous action chunks; pretraining standardizes delta EE motion + gripper. New action/state interfaces can be added during adaptation. | DDPM-style noise-prediction MSE on action chunks. | About 800k trajectories from 25 curated OXE datasets; target-robot demonstrations for fine-tuning. | Compute readout features, denoise a chunk with the lightweight action head (20 steps in the released setup), execute actions, and replan. |
| **[OpenVLA](../docs/Model_Zoo/Robotics/Policies/OpenVLA.md)** · [paper](https://arxiv.org/abs/2406.09246) | One RGB image + language instruction in the original model. | Prismatic VLM: fused DINOv2 + SigLIP visual features, MLP projector, Llama 2 7B. | One 7D action: delta EE translation/rotation + gripper; one token per dimension, 256 bins using 1st–99th percentile bounds. | Next-token CE on action outputs. | Pretrained Prismatic initialization; robot training on a curated 970k-episode OXE mixture; target demonstrations for adaptation. | Autoregressively produce seven action tokens, map bins back to continuous values, undo normalization, execute one step, and repeat. |
| **[π0](../docs/Model_Zoo/Robotics/Policies/Pi_0.md)** · [paper](https://arxiv.org/abs/2410.24164) | Two or three RGB views, instruction/subtask, and proprioceptive state. | PaliGemma (SigLIP + Gemma) + roughly 300M action expert; the two experts interact through layerwise attention. | Continuous 50-action chunks; embodiment-specific controls, padded to 18 dimensions in the paper. | Conditional FM velocity MSE; robot pre-training and task post-training use the same action objective. | About 10k hours of PI robot data across 7 configurations, mixed with OXE, Bridge V2, and DROID; curated task demonstrations for post-training. | Cache observation features; initialize a noisy chunk and perform 10 Euler updates; execute a prefix, then reobserve. No internal textual subtask generation in original π0. |
| **[FAST / π0-FAST](../docs/Model_Zoo/Robotics/Policies/Pi_0_FAST.md)** · [paper](https://arxiv.org/abs/2501.09747) | Policy: RGB views + instruction + discretized proprioceptive state. Tokenizer: continuous action chunk. | π0-FAST uses the PaliGemma VLM's autoregressive decoder; no separate flow action expert. FAST itself is normalization + DCT + quantization + BPE. | Variable-length token sequence encoding a roughly one-second continuous action chunk. | Policy: next-token CE on FAST tokens. Tokenizer: fit BPE merge rules to quantized coefficient sequences. | FAST+ tokenizer: about 1M real action chunks from multiple robots. Policy experiments: DROID/task data and the generalist π0 robot mixture; tokenizer and policy training are distinct. | Generate FAST tokens autoregressively; undo BPE, coefficient quantization, DCT, and normalization to recover a chunk; execute and replan. |
| **[π0.5](../docs/Model_Zoo/Robotics/Policies/Pi_0_5.md)** · [paper](https://arxiv.org/abs/2504.16054) | RGB views, overall instruction, and robot state; the low-level policy is conditioned on a predicted semantic subtask. | PaliGemma-based VLM + approximately 300M flow action expert; shared model supports semantic text and continuous control. | Training uses FAST tokens and, in post-training, continuous actions; deployment uses 50-action continuous chunks plus textual subtask outputs. | Pre-training: CE on FAST/text/grounding targets. Post-training: continued token CE + FM on action-labeled examples. | Mobile manipulation in about 100 homes, other home robots, cross-embodiment data including OXE, subtask/box labels, and web VLM data; post-training adds verbal subtask demonstrations and emphasizes successful mobile behavior. | Generate a subtask autoregressively, then condition the action expert on it and perform 10 flow updates. Execute continuous actions; refresh semantic planning less often than low-level control. |
| **[SmolVLA](../docs/Model_Zoo/Robotics/Policies/SmolVLA.md)** · [paper](https://arxiv.org/html/2506.01844v1) | RGB views + language + projected robot state. | Truncated SmolVLM2 (SigLIP + SmolLM2) + action expert with alternating cross-attention and causal self-attention; about 450M total parameters. | Continuous 50-action chunks in the target robot's action space. | FM velocity MSE; the reported recipe freezes the VLM and trains the action expert. | Community pretraining: 481 LeRobot datasets, 22.9k episodes, 10.6M frames; subsequent task-specific fine-tuning. | Encode context, refine the chunk with 10 flow steps, and execute/replan. Optional asynchronous inference overlaps prediction of the next chunk with execution. |
| **[RDT-1B](../docs/Model_Zoo/Robotics/Policies/RDT_1B.md)** · [paper](https://arxiv.org/html/2410.07864v1) · [project](https://rdt-robotics.github.io/rdt-robotics/) | Two-frame, three-camera RGB history + language + proprioception + control frequency. | Frozen SigLIP and T5-XXL encoders + 1.2B Robotics Diffusion Transformer with alternating image/language conditioning. | Continuous 64-action chunks in a physically interpretable 128D unified space, with masks for unavailable dimensions; mapped back to the robot's controls. | Diffusion denoising MSE predicting the **clean action chunk**, rather than noise, with invalid dimensions masked. | Pre-training: 1M+ episodes from 46 multi-robot datasets. Fine-tuning: 6k+ ALOHA bimanual demonstrations. | Encode conditions, initialize Gaussian noise, use 5 DPM-Solver++ steps to recover a chunk, select valid robot dimensions, and execute/replan. |
| **[GR00T N1](../docs/Model_Zoo/Robotics/Policies/GR00T_N1.md)** · [paper v2](https://arxiv.org/abs/2503.14734v2) | Current RGB views + instruction + proprioception; embodiment selects the state/action adapters. | Eagle-2 VLM features (SigLIP-2 + SmolLM2) condition a DiT through cross-attention; embodiment-specific MLP interfaces. | Continuous 16-action motor chunks; auxiliary video training can use continuous latent-action targets through a separate adapter. | FM velocity MSE + auxiliary target-object localization MSE. | Real robot data, human videos, physics simulation, and generated videos (about 8,376 pretraining hours in the v2 note); videos receive latent/IDM action labels. Post-training specializes to target-robot tasks. | Compute VLM features once; apply 4 flow updates to the 16-action chunk through the selected adapters; decode and execute. Video generators and action-labeling models are not needed at runtime. |
| **[WALL-OSS](../docs/Model_Zoo/Robotics/Policies/WALL_OSS.md)** · [paper v1](https://arxiv.org/abs/2509.11766v1) | Camera views + instruction + robot state; optional generated reasoning/subtask text conditions control. | Qwen2.5-VL-3B with shared self-attention, separate vision-language/action FFNs, and a continuous flow head. | FAST tokens in the Inspiration stage; continuous action trajectories in Integration/deployment; optional reasoning/subtask text. | Inspiration: VQA/text CE + FAST-token CE. Integration: FM for actions, with applicable semantic/VQA supervision; first fit the action branch, then train jointly. | Self-collected robot data + open datasets including DROID, BC-Z, Bridge, Agibotworld, and RH20T + general/embodied VQA; target-task fine-tuning where evaluated. | Optionally generate reasoning and a subtask, then generate continuous actions via the flow branch and refresh observations; a direct action path is also supported. The v1 paper does not specify a universal chunk horizon or solver-step count. |
