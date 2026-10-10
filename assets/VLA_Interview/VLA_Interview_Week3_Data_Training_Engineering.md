# Week 3 — Robot Data Pipeline & VLA Training Engineering

> **Track:** VLA Model Training Engineer interview preparation  
> **Prerequisite:** [Week 1 — Action Modeling](https://github.com/XP-PY/llm-prep-2026/blob/main/assets/VLA_Interview/VLA_Interview_Week1_Action_Modeling_Foundations.md), [Week 2 — Architecture Evolution](https://github.com/XP-PY/llm-prep-2026/blob/main/assets/VLA_Interview/VLA_Interview_Week2_Architecture_Evolution.md)  
> **Time budget:** 17–18 hours, 6 study days  
> **Format:** concepts → concrete tensor examples → runnable code → troubleshooting → interview drills  
> **Main learning goal:** Explain and implement the path from **raw robot demonstrations to a stable training update**, not just explain the architecture of π0.

**One-sentence mental model**

```text
Raw robot trajectories
    ↓ data audit, frame/time synchronization, coordinate semantics
Episode-aware training dataset
    ↓ observation history + future action chunk + valid masks
Train-only normalization + batch collation
    ↓ VLM features + state features + noisy continuous action
Flow-matching forward and masked loss
    ↓ backward, optimizer, scheduler, checkpoints
DDP/FSDP/ZeRO when needed
    ↓ evaluation and bad-case diagnosis
Improved dataset / objective / model
```

---

## 0. Week 3 Learning Outcomes

At the end, without looking at notes, you should be able to:

1. State what one **frame**, **step**, **episode**, **trajectory**, and **action chunk** mean in a robotics dataset.
2. Define a consistent convention for `observation[t]`, `state[t]`, and `action[t]`.
3. Explain how to synchronize camera, robot state, and action timestamps, and what happens if they are misaligned.
4. Distinguish **joint**, **end-effector**, **absolute**, **delta**, **position**, and **velocity** action representations.
5. Form one `[B, H, D]` action chunk without crossing an episode boundary or unintentionally crossing a subtask boundary.
6. Explain why **train/validation splitting comes before fitting normalization statistics**.
7. Explain padding masks, missing-DoF masks, camera masks, and why their meanings differ.
8. Show the actual tensors and calculations in a π0-style flow-matching training step.
9. Write a correct masked MSE loss and explain its denominator.
10. Explain AdamW, gradient accumulation, LR schedules, BF16, gradient clipping, checkpointing, and the order of calls.
11. Compare DDP vs FSDP2 vs DeepSpeed ZeRO-1/2/3.
12. Estimate parameter-side GPU memory and articulate what the estimate omits.
13. Troubleshoot **OOM**, **NaN**, training divergence, data-loader bottlenecks, and **low training loss but poor real-robot success**.
14. Present your **RM65 project** as a rigorous data-to-training pipeline.

> **Scope:** This week covers **training-engineering mechanics**. Robot runtime and deployment are Week 4; RL post-training and world models are Week 5.

---

## 1. Daily Plan (17–18 hours)

| Day | Focus | Time | Deliverable |
|---|---|---:|---|
| Day 1 | Episode schema, timestamp alignment, action semantics | 3 h | Write a data contract + detect 5 failure modes |
| Day 2 | Chunk sampling, masks, splits, normalization, LeRobot v3 | 3 h | Hand-calculate action windows; inspect a real batch |
| Day 3 | End-to-end conditional Flow Matching training step | 3.5 h | Execute companion training script; explain each tensor |
| Day 4 | Optimizer, gradients, schedules, mixed precision, memory | 2.5 h | Calculate effective batch and parameter memory |
| Day 5 | DDP/FSDP/ZeRO and debugging | 3 h | Draw communication flow + follow diagnosis runbook |
| Day 6 | RM65 project mapping + mock interview | 2.5–3 h | 10-min project walkthrough + code review + self-test |

**Reading policy:** If you already know a concept from `FSDP.md` or `Mixed_Precision_Training.md`, spend your time on how it changes a **VLA training loop**.

---

# Part I — From Robot Demonstrations to Training Samples

## 2. Write a Data Contract Before Writing `Dataset.__getitem__`

A robotics sample must define not only tensor shape but **physical meaning**.

Let an episode be

\[
\mathcal E_i = \{(t_k, I_k^{\mathrm{front}}, I_k^{\mathrm{wrist}}, q_k, a_k, \ell_k)\}_{k=0}^{T_i-1}.
\]

- \(t_k\): timestamp at control step \(k\).
- \(I_k\): synchronized camera observations.
- \(q_k\): robot proprioceptive state.
- \(a_k\): **command issued at this step** (our convention for this note).
- \(\ell_k\): task or subtask instruction available at this step.

**Critical convention:** If the robot logs `command[k]` and `measured_state[k+1]`, `command[k]` is the instruction that influences a future state. Do **not** silently relabel `measured_state[k+1]` as the command `a_k`. Whether a benchmark trains on current commands, next target positions, or state differences depends on its data contract.

### 2.1 Data-contract table (fill one for every new robot)

| Field | Example shape (per step) | Physical meaning / required metadata |
|---|---|---|
| `observation.images.front` | `[3, 480, 640]` | RGB, color order, camera intrinsics/extrinsics, timestamp |
| `observation.images.wrist` | `[3, 480, 640]` | Camera rigidly attached to which link? |
| `observation.state` | `[7]` | 6 joint angles in **rad**, gripper width in **m** or normalized command? |
| `action` | `[7]` | Position target, delta target, velocity, or next state? |
| `timestamp` | scalar | Monotonic seconds in a shared time base |
| `task` | string | Instruction actually available during deployment |
| `episode_id / frame_id` | scalar | Identifies sequence and ordering |
| `is_terminal` | scalar | End of episode / truncated recording |

**Important:** The shapes above are an illustration of a 6-DoF arm + one gripper; do not treat them as your actual RM65 configuration without checking.

### 2.2 Frame, timestep, episode, trajectory, chunk

- **Frame:** one image captured by a camera; cameras may run at different rates.
- **Control timestep:** one policy/controller decision interval.
- **Episode / trajectory:** one contiguous recorded rollout with start and stop boundaries. In code, these terms are often used interchangeably; a trajectory may also mean a state/action sequence extracted from an episode.
- **Subtask clip:** a labeled segment inside one episode. Its boundaries have semantic meaning.
- **Action chunk:** a future sequence \(A_t=[a_t,\ldots,a_{t+H-1}]\) supervised at one observation step.

---

## 3. Timestamp Alignment: The Silent Training Failure

Suppose the controller is sampled at 20 Hz:

\[
\Delta t=1/20=0.05\;\mathrm{s}.
\]

But the front camera records at 30 Hz, the wrist camera at 15 Hz, and the motor state at 100 Hz. Row indices alone do not correspond to the same physical time.

### 3.1 Recommended pipeline

```text
Raw source clocks and timestamps
    ↓ calibrate clock offsets / verify monotonicity
Choose a reference action/control time grid
    ↓
For each control time t:
  - camera: latest valid frame at or before t (causal)
  - state: nearest/interpolate only where justified, mindful of latency
  - action: the command issued for the chosen control interval
    ↓
Measure time residuals and reject/flag excessive gaps
    ↓
Aligned episode with explicit timestamps
```

**Example:** To build the observation for a command at time \(t=1.00\,\mathrm{s}\), using the nearest camera frame at \(t=1.03\) leaks information from *after* the action decision. A causal production policy should use the latest observation that was actually available before inference (e.g. \(0.97\,\mathrm{s}\)), accounting for capture, transport, and processing delays.

### 3.2 Alignment strategies

| Signal | Typical method | Risk |
|---|---|---|
| RGB frames | latest past frame, or explicitly timestamped nearest for offline analysis | future-frame leakage / staleness |
| Joint angles | interpolate synchronized measured samples when appropriate | invalid interpolation through wrap/discontinuities |
| Cartesian pose | translation interpolation + suitable rotation interpolation | naïve Euler-angle interpolation |
| Gripper state / discrete modes | zero-order hold | treating a discrete transition as continuous |
| Action commands | hold/associate with command interval | off-by-one command/observation pairs |
| Language instructions | join by task/subtask interval | label from future task stage |

### 3.3 Alignment audits to implement

- Is time monotonic inside each episode?
- How large is the image–state offset distribution (median, p95, max)?
- Any duplicated or dropped frames?
- Are there unreasonable joint-velocity spikes?
- Did the command target belong to the **same** control convention used in deployment?
- Are stage labels visible to the model only when they will be available at inference?

**Interview answer:** Timestamp alignment is not just video resampling: it is preserving the **causal relationship** between observation and commanded action.

---

## 4. Action Representation: Do Not Train Until You Can Explain `a_t`

Two models may both output `[B, 50, 7]` yet predict fundamentally different quantities.

### 4.1 Joint-space vs task-space

**Joint-space**

\[
a_t=q_t^{\mathrm{target}} \quad\text{or}\quad a_t=q_{t+1}^{\mathrm{target}}-q_t.
\]

**Task-space (EEF)**

\[
a_t=(\Delta p_t,\Delta R_t,g_t),
\]

where \(p\) is an end-effector position and \(R\in SO(3)\) an orientation. A 6D rotation representation used inside a model is **not** a six-component angular velocity.

### 4.2 Absolute vs relative targets

- Absolute position target: \(q_{\mathrm{cmd}}=a_t\).
- Relative joint target: \(q_{\mathrm{cmd}}=q_t+\Delta q_t\).
- Joint velocity target: \(\dot q_{\mathrm{cmd}}=a_t\), interpreted through the controller's update period and low-level control law.
- Relative end-effector translation in base frame: \(p_{\mathrm{cmd}}^{B}=p_t^{B}+\Delta p^{B}\).

**Coordinate-frame trap:** A delta in **base frame** is not numerically the same as a delta in **tool frame**. If the end-effector rotates, the direction of a tool-frame delta changes in the base frame:

\[
\Delta p^B=R_{BT}\Delta p^T.
\]

### 4.3 Units and normalization

Joint angle in rad, EEF translation in m, gripper width in mm, and normalized gripper command in `[0,1]` cannot be mixed without metadata.

**Do not assume all VLA models normalize identically.** Diffusion Policy-style models often use per-dimension normalized actions; RDT explicitly emphasizes physically interpretable quantities/units and does not simply apply per-dataset `[-1, 1]` normalization to everything. A pretrained model's processor, statistics, and action semantics are part of its interface.

### 4.4 Small action-semantics quiz

> Recorded `state[t] = q_t` and `action[t] = q_{t+1}` (position command). Is `action[t] - state[t]` an absolute action or delta action? Can a model trained on `q_{t+1}` be deployed by directly adding its output to the current joint angle?

**Answer:** The difference is a delta, whereas the logged target is absolute. Adding an already-absolute target to \(q_t\) would be a serious control error.

---

# Part II — Episode-Aware Dataset, Chunks, Masks, Statistics

## 5. Generate a Training Window

A VLA sample may be written as:

\[
X_t=(O_{t-H_o+1:t},\ell_t,s_t),\qquad
Y_t=A_t=(a_t,\ldots,a_{t+H_a-1}).
\]

Where:

- \(H_o\): observation history horizon;
- \(H_a\): future action prediction horizon;
- \(H_e\): runtime **execution** horizon (not necessarily equal to \(H_a\)).

### 5.1 Shapes

Assume **illustrative** values: `B=4`, `H_o=2`, `H_a=16`, `D_s=7`, `D_a=7`, 2 cameras at `256×256`.

| Tensor | Shape |
|---|---|
| Front images | `[4, 2, 3, 256, 256]` |
| Wrist images | `[4, 2, 3, 256, 256]` |
| Robot state | `[4, 7]` |
| Language IDs | `[4, L]` |
| Action chunk | `[4, 16, 7]` |
| Valid-step mask | `[4, 16, 1]` |
| DoF availability mask | `[4, 1, 7]` |
| Flow time | `[4, 1, 1]` |
| Gaussian action noise | `[4, 16, 7]` |

**Causal rule:** Observations must come from \(\leq t\); supervised actions may come from \(\geq t\) because they are *targets*, not future conditioning inputs.

### 5.2 Boundary handling

For an episode with `T=20`, chunk horizon `H_a=8`, and a training anchor `t=18`:

```text
actual future actions:  a18, a19
right padding:           PAD, PAD, PAD, PAD, PAD, PAD
valid-step mask:         1,   1,   0,   0,   0,   0,   0,   0
```

You may instead **drop** incomplete windows. Trade-off:

- Drop: simpler, but biases toward interior timesteps and wastes data.
- Pad + mask: preserves more anchors, but demands correct masking.

**Never** let the remaining positions accidentally read the beginning of the next episode.

### 5.3 Subtask boundaries (specific to your RM65 project)

If a chunk starts in `reach` and crosses into `release`, there are two legitimate design choices:

1. **General end-to-end policy:** supervise across stage boundaries if the policy is supposed to smoothly model the whole task.
2. **Plan-conditioned subtask policy:** truncate the chunk at the current subtask boundary if it should predict actions only for the selected Plan.

Neither is universally correct. Your project uses the second idea. You should explain the architectural reason and its downside (short valid chunks near a boundary, loss-weighting and transition behavior).

---

## 6. Four Masks You Must Keep Separate

| Mask | Purpose | Example |
|---|---|---|
| `time_valid_mask` | Ignore action timesteps beyond clip end | `[B,H,1]` |
| `dof_mask` | Ignore non-existent/unlabeled action dimensions | `[B,1,D]` |
| `camera_mask` | Denote camera modality missing | `[B,N_cam]` |
| `attention_mask` | Control token visibility in Transformer | `[B,N,N]` or implementation-specific |

A tensor value of **0** can mean a legitimate physical zero. Therefore `value == 0` must not be used to infer padding.

For action error \(E=(\hat v-v^*)^2\) with \(E\in\mathbb R^{B\times H\times D}\), combine valid-step and DoF masks:

\[
M_{b,h,d}=M^{\mathrm{time}}_{b,h,1}\cdot M^{\mathrm{dof}}_{b,1,d}.
\]

A valid-element mean loss is:

\[
\boxed{\mathcal L=\frac{\sum_{b,h,d}M_{b,h,d}E_{b,h,d}}{\max(1,\sum_{b,h,d}M_{b,h,d})}}.
\]

**Important implementation detail:** `M` must be broadcast/expanded to the full `[B,H,D]` error shape **before** you count its elements. Dividing by a `[B,H,1]` mask sum without accounting for `D` multiplies your loss scale by the action dimension.

```python
squared_error = (pred_velocity - target_velocity).square()  # [B,H,D]
valid = (time_mask * dof_mask).expand_as(squared_error)
loss = (squared_error * valid).sum() / valid.sum().clamp_min(1)
```

---

## 7. Train/Validation Splitting and Leakage

**Wrong:** Flatten 442 trajectories into individual frames, then random-split all frames into train and validation. Neighboring frames of the same demonstration will appear on both sides; validation loss can look excellent even when generalization is poor.

**Baseline:** Split at **episode level** before frame/chunk extraction.

**Stricter evaluation:** Depending on your claims, also hold out environments, objects, instructions, operators, or task compositions. “Generalize to unseen objects” requires objects to be held out, not merely episodes.

Suggested setup for your project (a **proposal**, not a claim about your actual split):

- Training: ~80% of episodes.
- Validation: ~10%.
- Test: ~10% kept frozen.
- Where possible, stratify by task/instruction while preventing identical recording sessions from leaking.
- Evaluate the final real-robot rollout separately.

**Data-leakage checklist**

- [ ] Split by episode before generating overlapping chunks.
- [ ] Check near-duplicate sessions / repeated trajectories.
- [ ] Fit all action/state normalization statistics on train only.
- [ ] Avoid using validation set for automatic data filtering choices without disclosing it.
- [ ] Keep test scenarios untouched while tuning.
- [ ] Do not use future Plan/Box labels as model conditioning at evaluation unless the deployed system predicts them.

---

## 8. Action Normalization

Two common choices:

### 8.1 Standardization

\[
\tilde a_d=\frac{a_d-\mu_d}{\sigma_d+\varepsilon},
\qquad
a_d=\tilde a_d\sigma_d+\mu_d.
\]

### 8.2 Min–max scaling

\[
\tilde a_d=2\frac{a_d-a_d^{\min}}{a_d^{\max}-a_d^{\min}}-1.
\]

Robust quantile-based methods can reduce the impact of outliers.

### 8.3 Production considerations

- Compute statistics on **train-only, observed, valid physical components**.
- Do not count padded zeros toward means or stds.
- If a dimension is constant, guard near-zero standard deviation.
- Choose **global vs per-robot** statistics deliberately; mixing joint and EEF semantics may require separate adapters.
- Save the **normalization statistics in the checkpoint**. Without them, the predicted action cannot reliably be decoded to physical commands.
- Match training normalization and inference denormalization *exactly*.

**Interview trap:** More GPUs or a better optimizer will not fix actions sent in wrong units or wrong coordinate frames.

---

## 9. LeRobot Dataset v3: What to Know (2026)

The current LeRobot dataset stack exposes **episode-aware access** while storing many episodes in relatively large files. The main idea is to separate user-facing episode indexing from underlying Parquet/video shards.

High-level components:

```text
LeRobot dataset
├── meta/        # feature schema, task/episode metadata, stats
├── data/        # timestamped state/action rows in Parquet shards
└── videos/      # multi-camera recordings stored in MP4 shards
```

Exact metadata filenames and APIs are version-dependent; use the installed LeRobot version's `info.json` schema and official docs, rather than assuming an old one-episode-per-file convention.

A practical API:

```python
from lerobot.datasets import LeRobotDataset
from torch.utils.data import DataLoader

fps = 20  # illustrative; fetch actual fps from dataset metadata
H = 16

delta_timestamps = {
    "observation.images.front": [-1 / fps, 0.0],
    "action": [i / fps for i in range(H)],
}

dataset = LeRobotDataset(
    "your-account/your-vla-dataset",
    delta_timestamps=delta_timestamps,
)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

batch = next(iter(loader))
print(batch.keys())
print(batch["action"].shape)  # expected [8,16,D] for matching schema
```

**Do not copy these example feature keys blindly:** inspect `dataset.meta.features` for your actual camera name(s), states, `fps`, and compatible temporal offsets.

**What to remember for interviews:** LeRobot helps with dataset schema, temporal windows, video decode, and model training integration; it does **not** automatically solve the physical meanings of action coordinates and calibration.

References: [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), [LeRobot Dataset API](https://huggingface.co/docs/lerobot/main/api/datasets), [Porting to v3](https://huggingface.co/docs/lerobot/main/porting_datasets_v3).

---

# Part III — A Real Training Step, Not Just a Formula

## 10. A π0-Style Conditional Flow-Matching Step

A sample batch contains observation \(o\), robot state \(s\), and normalized **clean** future action chunk \(A\in\mathbb R^{B\times H\times D}\).

Draw independent random variables:

\[
\tau_b\sim U[0,1],\quad \epsilon\sim\mathcal N(0,I).
\]

Make a noisy/interpolated action:

\[
A^\tau=(1-\tau)\epsilon+\tau A.
\]

Target velocity:

\[
\boxed{u=A-\epsilon.}
\]

Condition the action expert on encoded observations, state, noisy actions and the **flow time** (not the physical robot timestep):

\[
\hat u=v_\theta(A^\tau,\tau,\mathrm{VLM}(I,\ell),s).
\]

Loss (masked if necessary):

\[
\boxed{\mathcal L_{\mathrm{FM}}=\frac{\sum M\odot(\hat u-u)^2}{\sum M}.}
\]

### 10.1 What changes in the π0 backbone?

```text
RGB images ──► Vision encoder ──┐
Language ────► Tokenizer / LLM ─┤──► pretrained VLM context
                                │             │
Robot state ──► state projector ─┼─────────────▼
Noisy actions ► action projector ┼────► action expert
Flow time ────► time embedding ──┘             │
                                               ▼
                                        velocity [B,H,D]
                                               │
                                        masked MSE vs A−eps
                                               │
                                        backward / optimizer
```

The **mathematics** is simple. The **training-engineering problem** lies in making each conditioning tensor correct and the gradients stable.

### 10.2 Minimal correct flow loss

```python
# batch["actions"]: [B,H,D], already normalized
A = batch["actions"].to(device)
B, H, D = A.shape

tau = torch.rand(B, 1, 1, device=device)
noise = torch.randn_like(A)
x_tau = (1 - tau) * noise + tau * A
velocity_target = A - noise

velocity_pred = policy(
    images=batch["images"].to(device),
    state=batch["state"].to(device),
    language=batch["language"],  # tokenize/collate in your actual model
    noisy_actions=x_tau,
    flow_time=tau,
)

valid = batch["action_mask"].to(device).expand_as(A)
sq_err = (velocity_pred.float() - velocity_target.float()).square()
loss = (sq_err * valid).sum() / valid.sum().clamp_min(1)
```

**Note:** This is an interface sketch: a real π0 policy API, image transforms, language processing, token masks, and actions may have different names.

---

## 11. Training vs Inference: Identify Label Leakage

| Quantity | Training | Inference |
|---|---|---|
| Camera images / language / state | inputs | inputs |
| Ground-truth future actions \(A\) | supervision, to build \(A^\tau\) | **not available** |
| Gaussian noise \(\epsilon\) | sampled | sampled initial action noise |
| Flow time \(\tau\) | random sample | numerical solver steps |
| Target velocity \(A-\epsilon\) | supervised label | **not available** |
| Predicted Plan | possible supervision/conditioning | predicted by model |
| Ground-truth Plan | may be training label | **not available** unless externally provided |
| Future annotated Box | supervised label | **not available** unless separately predicted |

**Key principle:** If the action branch attends to ground-truth FAST/Box/Plan tokens during training, but these are not present at runtime, the training graph can exploit information that disappears during inference. You explicitly addressed this issue by constraining attention visibility in your project.

---

## 12. Loss Masks and Auxiliary Heads

Suppose you jointly train action, plan, box, and discrete FAST outputs:

\[
\mathcal L=\lambda_A\mathcal L_A
+\lambda_P\mathcal L_P
+\lambda_B\mathcal L_B
+\lambda_F\mathcal L_F.
\]

- \(\mathcal L_A\): masked flow MSE over valid chunk timesteps and action DoFs.
- \(\mathcal L_P\): token cross-entropy for Plan labels, masked for missing Plan supervision.
- \(\mathcal L_B\): regression or coordinate-token loss **only at annotated keyframes**, with correct camera/box validity.
- \(\mathcal L_F\): cross-entropy on FAST action tokens, mask padding tokens.

### 12.1 Wrong loss normalization

```python
loss = (all_action_sq_error * valid).mean()
```

This divides by **all padded elements**, so batches with more padding appear to have smaller loss. This changes learning pressure as sequence lengths vary.

### 12.2 Another subtlety: multi-task loss scale

A raw `CE ≈ 3.0` and `FM MSE ≈ 0.1` **do not imply** the CE task's gradients are 30 times larger. Check gradients and learning signals; tune auxiliary weights using controlled ablations, not numeric loss magnitude alone.

---

## 13. Fully Runnable Companion Exercise

Companion source: **`week3_minimal_flow_train.py`** (download together with this Markdown).

Run:

```bash
python -m pip install torch  # only if PyTorch is not installed
python week3_minimal_flow_train.py
```

It creates synthetic episodes and demonstrates, end-to-end:

```text
independent episodes
→ episode-level split
→ train-only action statistics
→ future action chunks
→ mask incomplete chunk suffix
→ noisy interpolation
→ conditional MLP predicts velocity
→ masked MSE
→ backward + gradient clipping + AdamW
→ held-out validation loss
→ checkpoint contains normalization stats
```

**What is simplified:** The script uses already-encoded `[OBS_DIM]` features instead of actual RGB images and a giant VLM, and it simulates trajectories rather than controlling the RM65. Its purpose is to make the data/shape/loss/optimizer chain completely transparent. It is **not** an official π0 implementation.

### 13.1 Expected minimal console behavior

Output has three epochs of `train_loss` and `valid_loss`, then a message that smoke tests passed and a checkpoint was saved. The exact floating-point values depend on hardware/precision.

### 13.2 What you must inspect

- `Episode`: state/action association and episode ownership.
- `ChunkDataset.__getitem__`: how the last chunk is padded and masked.
- `smoke_tests`: why the last frame of an episode must have **exactly one** valid future action under our convention.
- `flow_loss`: whether tensor shapes are `[B,H,D]` and denominator counts **valid elements**.
- `optimizer`: `zero_grad`, `backward`, `clip_grad_norm_`, `step` order.
- `checkpoint`: normalization stats must be preserved for inference.

### 13.3 Extension tasks (code, not theory)

- [ ] Add a `dof_mask` with one inactive action dimension and confirm masked loss is unchanged when that dimension's target is corrupted.
- [ ] Add `subtask_end[t]`, truncate the chunk at the current subtask end, and verify boundary masking.
- [ ] Replace z-score normalization with quantile normalization; retain its inverse parameters.
- [ ] Add `gradient_accumulation_steps=4` and check the effective batch.
- [ ] Log learning rate and gradient norm at every optimizer step.
- [ ] Add deterministic synthetic camera features to the observation; do not allow images from `t+1` as input for action at `t`.

---
# Part IV — Optimization and Memory

## 14. Understand the Training Loop Order

The standard single-device update is:

```text
get batch
→ move tensors to device
→ forward under optional autocast
→ compute loss
→ backward
→ optionally clip gradients
→ optimizer.step()
→ scheduler.step()   # usually once per optimizer update
→ zero_grad(...)
```

There are two common valid placements for `zero_grad()`:

- Before each new update (as in the companion script).
- Immediately after finishing the preceding update.

What matters is that gradients **must not accidentally accumulate between optimizer updates**.

### 14.1 AdamW in one equation

Adam maintains moving moments:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
\]

With bias-corrected moments, AdamW's decoupled weight decay can be expressed as:

\[
\boxed{\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}}.
\]

- **Adam** adapts step size per parameter based on gradient history.
- **AdamW** applies weight decay separately from the adaptive gradient update.
- Use deliberate optimizer parameter groups: bias and normalization parameters are often excluded from weight decay, according to the chosen training recipe.

Existing notes: [AdamW](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Optimizer/AdamW.md).

---

## 15. Effective Batch Size & Gradient Accumulation

Define:

- \(b\): microbatch size **per GPU**;
- \(g\): number of data-parallel ranks / GPUs;
- \(k\): accumulation steps before each optimizer update.

\[
\boxed{B_{\mathrm{global}}=b\times g\times k.}
\]

**Example:** `per_device_batch=2`, `world_size=4`, `grad_accum=8`:

\[
B_{\mathrm{global}}=2\times4\times8=64.
\]

An **optimizer step** occurs after 8 microbatches per GPU, not after every microbatch.

### 15.1 Correct scaling for fixed equal-sized microbatches

```python
optimizer.zero_grad(set_to_none=True)
for micro_idx in range(accum_steps):
    loss = model_loss(next(microbatches))
    (loss / accum_steps).backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
```

**Why divide by `accum_steps`?** Otherwise the accumulated gradient becomes approximately \(k\) times larger (for equal-size mean losses), changing the effective learning rate.

### 15.2 Subtle issue: unequal valid-element counts

Suppose microbatch A has 14 valid action elements, B has 70. Averaging their **per-microbatch means** gives each microbatch equal weight, whereas averaging over all 84 valid elements weights B more strongly.

For exact valid-element weighting, the goal is:

\[
\mathcal L=\frac{\sum_{j=1}^k\mathrm{errorSum}_j}
{\sum_{j=1}^k\mathrm{validCount}_j}.
\]

For distributed training, the denominator also needs to reflect the desired **global** reduction semantics. Design your loss reduction intentionally, especially when subtask truncation creates variable valid lengths. The companion script uses per-batch valid-element normalization; it does not implement gradient accumulation.

### 15.3 Common interview traps

- Calling `scheduler.step()` on every microbatch even though the schedule is defined per **optimizer update**.
- Forgetting `loss /= accum_steps` in equal-size accumulation.
- Treating 4 GPUs as automatically providing 4 times more memory to each DDP replica (they do not).
- Assuming increasing gradient accumulation makes activations for each microbatch smaller (only smaller **microbatch size** does that).

---

## 16. Learning Rate, Warmup, Gradient Clipping, EMA

### 16.1 Learning-rate warmup

A simple linear warmup over \(W\) optimizer updates:

\[
\eta(s)=\eta_{\max}\min\left(1,\frac{s}{W}\right).
\]

Why useful: pretrained backbone and randomly initialized action head can have very different initial gradient scales. Warmup may help avoid a large early update, but **not every paper uses warmup**; choose it based on the training recipe and validation.

### 16.2 Gradient clipping

For an unclipped gradient \(g\), global-norm clipping uses:

\[
\tilde g=g\cdot\min\left(1,\frac{c}{\|g\|_2}\right).
\]

- Apply clipping **after gradients are accumulated** and **before optimizer step**.
- With FP16 `GradScaler`, call `scaler.unscale_(optimizer)` before clipping.
- Record gradient norms; clipping every step may indicate a deeper numerical issue.

### 16.3 EMA (exponential moving average)

Some diffusion/robot policies maintain smoothed model weights:

\[
\theta_{\mathrm{EMA}}\leftarrow\beta\theta_{\mathrm{EMA}}+(1-\beta)\theta.
\]

EMA is not the same as Adam momentum, and EMA should be treated as a **separate copy of weights** for evaluation/inference when used. It consumes memory and checkpoint space.

---

## 17. Mixed Precision: FP32, FP16, BF16

| dtype | Typical storage | Relevant property |
|---|---:|---|
| FP32 | 4 bytes/value | broad numerical range and more precision |
| FP16 | 2 bytes/value | limited exponent range; gradients may under/overflow |
| BF16 | 2 bytes/value | FP32-like exponent range; usually no gradient scaling required |

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = compute_loss(model, batch)
loss.backward()
optimizer.step()
```

- Use BF16 autocast only if the device supports it and the model/operations behave correctly.
- Autocast does **not** mean every tensor, optimizer state, and model weight is stored in BF16.
- For FP16 AMP, gradient scaling is often necessary; for BF16, usually not.
- Consider computing sensitive loss reductions in FP32 (`.float()`).

Your existing deep dive: [Mixed Precision Training](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Training_Optimization/Mixed_Precision_Training.md).

---

## 18. Estimating GPU Memory

For training a model with \(P\) trainable parameters using one common **mixed-precision Adam** accounting convention:

| Component | Illustrative bytes/param |
|---|---:|
| BF16 weights | 2 |
| BF16 gradients | 2 |
| FP32 master parameter copy | 4 |
| Adam FP32 first/second moments | 8 |
| **Total state estimate** | **16** |

\[
\boxed{M_{\mathrm{states}}\approx16P\;\mathrm{bytes}.}
\]

For `3.3B` fully trainable parameters:

\[
3.3\times10^9\times16 \approx 52.8\;\mathrm{GB}
\]

**This is only a rough model-state example**. Some BF16 frameworks keep FP32 parameter storage and do not have the exact `2+2+4+8` layout, while gradient/optimizer sharding or 8-bit optimizers change the accounting.

**It excludes:**

- activations saved for backward;
- visual tokens and long sequence activation tensors;
- temporary kernel workspaces;
- attention caches/buffers;
- FSDP all-gather / communication buffers;
- data prefetching;
- allocator fragmentation.

Therefore, a `24 GB` card cannot be evaluated by parameter count alone.

### 18.1 Why image tokens matter

Suppose the VLM sees 3 cameras. Increasing image resolution can dramatically increase visual token count and sequence/activation memory. The same robot policy with different image processors can have very different training memory.

### 18.2 Activation checkpointing

```text
normal forward: save many activations
checkpointed forward: save only selected boundaries
backward: recompute missing activations
```

Trade-off: **lower activation memory, higher compute**.

Your existing notes: [Memory Estimation](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Math/Memory_Estimation.md) and [Gradient Checkpointing](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Training_Optimization/Gradient_Checkpointing.md).

---

# Part V — Distributed Training and Debugging

## 19. DDP: Multiple Full Models, Different Microbatches

For \(G\) GPUs, rank \(r\) computes a local gradient \(g_r\). DDP all-reduces gradients so the model replicas remain synchronized:

\[
\boxed{g=\frac{1}{G}\sum_{r=1}^Gg_r.}
\]

```text
GPU0: full model + local data0 ─► local grads0 ┐
GPU1: full model + local data1 ─► local grads1 ├─► all-reduce
GPU2: full model + local data2 ─► local grads2 ┤     │
GPU3: full model + local data3 ─► local grads3 ┘     ▼
                                         each rank applies same update
```

DDP is excellent when a model already fits on each GPU; it does **not** shard the full parameters/optimizer states automatically.

### 19.1 DDP implementation skeleton

```python
# Launch: torchrun --standalone --nproc_per_node=2 train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=2, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # important for new shuffle order each epoch
    for batch in loader:
        # forward → loss → backward → step
        ...

dist.destroy_process_group()
```

**This is an illustrative skeleton**, not a runnable drop-in script (`MyModel`, `dataset`, and `num_epochs` must be defined). Device mapping, correct sampler behavior, save-on-rank-0, and step-count synchronization must be handled in real code.

### 19.2 DDP + gradient accumulation

With `accum_steps=4`, use `no_sync()` for the **first three forward/backward pairs** and normal sync for the last. The `forward` must also occur inside `no_sync()` to skip synchronization correctly.

```python
from contextlib import nullcontext

optimizer.zero_grad(set_to_none=True)
for i in range(accum_steps):
    context = model.no_sync() if i < accum_steps - 1 else nullcontext()
    with context:
        loss = compute_loss(model, next_microbatch())
        (loss / accum_steps).backward()
optimizer.step()
```

This simplified pattern assumes equal per-microbatch loss weighting and is not a complete mixed-precision/mask-weighted solution.

**References:** [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

---

## 20. FSDP2: Shard Model States, Materialize Around Compute

DDP memory issue: every rank holds the full model states. FSDP2 uses `torch.distributed.fsdp.fully_shard` to shard parameters/gradients/optimizer states across data-parallel ranks, with **all-gather** when full parameters are needed and **reduce-scatter** for gradient handling.

```text
Between layers / outside compute:
 GPU0: weight shard A
 GPU1: weight shard B
 GPU2: weight shard C
 GPU3: weight shard D

For current layer forward/backward:
         all-gather required parameter shards
                     ↓
              compute layer
                     ↓
             reshard as appropriate
                     ↓
           reduce-scatter gradients
```

**Trade-off:** less per-rank model-state memory vs more communication and transient unsharded buffers.

### 20.1 FSDP2 pseudocode (conceptual)

```python
from torch.distributed.fsdp import fully_shard

# after process-group setup and device placement:
for block in model.transformer_blocks:
    fully_shard(block)  # shard bottom-up for better overlap
fully_shard(model)
```

Actual API constraints depend on model structure, device mesh, mixed precision policy, and checkpoint format. See the [official FSDP2 API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html) and your [existing FSDP note](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Parallelism/FSDP.md).

**Question:** Does having `2 × 24 GB` under FSDP mean one model now has an unrestricted `48 GB` memory budget? **No.** Activations, temporary all-gathered layers, communication buffers, and per-rank local microbatch memory still consume VRAM; sharding is not a free pooling of VRAM.

---

## 21. DeepSpeed ZeRO Stages

| Strategy | Optimizer states | Gradients | Model parameters |
|---|---|---|---|
| Plain DDP | replicated | replicated | replicated |
| ZeRO-1 | **sharded** | replicated | replicated |
| ZeRO-2 | **sharded** | **sharded** | replicated |
| ZeRO-3 | **sharded** | **sharded** | **sharded** |
| Full-shard FSDP | **sharded** | **sharded** | **sharded** |

> This table is about **model states**, not activation memory. The exact memory/communication behavior also depends on offload, bucketization, precision, and chosen wrapping strategy.

### 21.1 Which strategy would you start with?

- **Model comfortably fits one GPU:** DDP, because it is simpler.
- **Model states dominate VRAM:** try FSDP2 or ZeRO-3.
- **Optimizer states/gradients are the bottleneck but full weights fit:** ZeRO-1/2 may be sufficient.
- **Activation memory dominates:** smaller local microbatches, shorter token budgets, selective checkpointing, attention optimizations.
- **Small dataset / limited GPUs:** frozen backbone + trainable action expert/LoRA may be more efficient than full fine-tuning.

Official resource: [DeepSpeed ZeRO tutorial](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/zero.md).

---

## 22. Gradient Accumulation vs DDP vs FSDP: Three Different Levers

| Technique | Primarily solves | Does **not** automatically solve |
|---|---|---|
| Gradient accumulation | larger effective batch under smaller microbatch | model weights too large for device |
| DDP | data-parallel throughput | parameters replicated on every GPU |
| FSDP/ZeRO-3 | per-GPU model-state memory | activations for each local microbatch |
| Activation checkpointing | activation memory | optimizer-state memory |
| BF16 mixed precision | throughput / many tensor buffers | all FP32 optimizer states |
| LoRA / frozen VLM | reduce trainable states and gradients | cost of loading base model and activations |

**Memorize this classification; it is more valuable in an interview than vague “add GPUs / reduce batch size” advice.**

---

## 23. Debugging Runbook: Training Engineering

### 23.1 CUDA OOM

**First ask: when does OOM occur?**

| Phase | Likely contributors | First actions |
|---|---|---|
| Model load | weights, duplicated devices, wrong dtype | inspect dtype, rank-device mapping |
| Forward | image tokens, activations, input resolution, batch | reduce microbatch, visual tokens, sequence length |
| Backward | saved activations, gradients, temporary buffers | gradient checkpointing, frozen backbone, smaller microbatch |
| Optimizer step | Adam optimizer state initialization | shard optimizer or freeze parameters |
| Only after many steps | leaks from retaining graphs/tensors | check `.append(loss)` without `.detach()` and stale references |
| FSDP communication | gathered layer buffers / bucket config | adjust wrapping, bucket/prefetch, peak instrumentation |

**Basic instrumentation**

```python
torch.cuda.reset_peak_memory_stats()
# run one training step
print(torch.cuda.max_memory_allocated() / (1024**3), "GiB allocated")
print(torch.cuda.max_memory_reserved() / (1024**3), "GiB reserved")
```

> Reserved memory is the allocator's cache, not the same thing as currently allocated live tensors. Measure rather than guessing.

### 23.2 NaN or Inf loss

Investigate in order:

1. Is the **raw dataset** finite? (`torch.isfinite` on state/actions, no invalid rotation conversions)
2. Is normalization stable? (`std > 0`, no absurd outliers)
3. Does the loss divide by a nonzero valid-element count?
4. Are you masking only legitimate valid elements?
5. Are BF16/FP16 and the optimizer configured sensibly?
6. Is learning rate too high or gradient norm exploding?
7. Does the numerical issue appear before or after one optimizer update?
8. Do all ranks participate consistently in distributed collectives?

**Fast diagnosis:** Temporarily train a tiny model on 1–4 batches in FP32, with fixed seeds and anomaly detection if necessary. This isolates data/logic errors from scale issues.

### 23.3 Loss decreases, robot still fails

Separate four types of failure:

| Failure class | Clues | What to test |
|---|---|---|
| Data semantics | consistent wrong pose/direction | units, coordinate frames, action target convention |
| Grounding | picks wrong object | camera/viewpoint, language and object labels, box prediction |
| Action dynamics | correct target but poor approach/grasp | control delay, normalization, action horizon, low-level controller |
| Generalization | only unseen objects/scenes fail | split, diversity, augmentation, OOD coverage |

**Offline action loss and robot task success measure different things.** A small MSE does not guarantee physically valid contacts, correct long-horizon task sequencing, or recovery from errors.

### 23.4 Training stalls or GPU utilization is low

Possible culprits:

- video decoding / I/O bottleneck;
- unbalanced DDP ranks / slowest-rank waits;
- too many small CPU→GPU transfers;
- CPU augmentation dominates;
- many tiny kernel launches;
- FSDP all-gather bandwidth bottlenecks;
- frequent checkpoint saving/evaluation;
- non-pinned memory, insufficient workers, poorly configured prefetch;
- CPU utilization or storage throughput limits.

**Measure:** data-load time, forward time, backward time, optimizer time, GPU utilization, communication time separately. `nvidia-smi` alone is not a complete profiler.

---

## 24. Checkpoints and Reproducibility

A proper training checkpoint should contain at least:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler else None,
    "global_step": global_step,
    "epoch": epoch,
    "config": training_config,
    "normalization": normalization_stats,
    "dataset_version": dataset_version,
    "model_arch_version": model_arch_version,
    # RNG and sampler states if exact reproducible resume is required
}
```

If using distributed/FSDP training, state-dict saving/loading may be **sharded**, requiring distributed checkpoint-aware utilities. You cannot assume plain `torch.save(model.state_dict())` is appropriate for every FSDP configuration.

### 24.1 Model evaluation checkpoints

Do not save solely based on training loss. For real robots, consider:

- train/val action or flow loss;
- task success rate on fixed scenarios;
- success by instruction/object/scene;
- latency and throughput;
- safety incidents / joint-limit violations;
- ablation settings and random seeds.

---

# Part VI — Your RM65 Project: Data-to-Training Narrative

## 25. Bring the Week Back to Your Own Experience

You reported that your RM65 desktop-arm project includes:

- **442** teleoperated trajectories;
- **239,641** frames;
- **52** language instructions;
- wrist and overhead RGB, joint/gripper states/actions;
- **1,892** audited subtask clips and per-frame Plan labels;
- **1,450** annotated keyframes and **2,900** camera-level annotations;
- ACT/SmolVLA/π-family baselines, planning-conditioned variants, and real-robot evaluation.

**These are project details you provided, not numbers independently audited by this file.**

### 25.1 Recommended 90-second technical explanation

> We collected multi-camera robot demonstrations and first established episode/timestamp/action semantics. We derived subtask segments using gripper events and task instructions, then audited boundaries and removed abnormal trajectories. We generated per-frame Plan labels and sparse keyframe object annotations. For model training, we formed observation/state and future action chunks, normalized actions according to the chosen policy interface, and used masks so action supervision stayed inside valid subtask boundaries. We extended a π0.5-style policy with explicit Plan conditioning and auxiliary Box/FAST targets, while the continuous action branch used Flow Matching. We compared against standard policies and evaluated on the real RM65 platform. The key engineering challenge was making stage-level semantic supervision compatible with low-level continuous action learning without training-time label leakage.

**Replace any sentence above with your actual implementation details if they differ.** A model-training interview will quickly expose details that were not truly implemented.

### 25.2 Follow-up questions likely to target your project

1. **Your action space:** Are you predicting absolute joint targets, deltas, or EEF commands? What are the units?
2. **Timestamp alignment:** How did wrist/front camera, states, and commands line up in time?
3. **Plan segmentation:** Why use gripper state for segmentation? What cases break the heuristic?
4. **Transition boundaries:** Why truncate action chunks at subtask ends? How do you handle masks and short chunks?
5. **Sparse boxes:** How do you ensure only labeled keyframes contribute to Box loss?
6. **FAST branch:** How many FAST tokens per sample? Did FAST loss help the flow branch empirically?
7. **Attention masking:** Can action tokens attend to ground-truth Plan/Box/FAST targets? Why or why not?
8. **Train/inference mismatch:** At training time do you condition on predicted Plan, ground-truth Plan, or a mixture? How do you mitigate wrong-Plan cascading errors?
9. **Normalization:** Are stats trained per robot/task? Is the inverse applied before control?
10. **Validation:** How did you split trajectories to avoid nearly identical frames in train/validation?
11. **Failures:** When a task fails, how do you attribute the cause to Plan, perception, control, or data?
12. **Compute:** Which modules did you freeze? What DDP/FSDP strategy, batch size, optimizer, and learning rate did you really use?
13. **Metric:** What exactly is your real-robot success criterion? Number of rollouts? Same scene distribution for baselines?
14. **Claim discipline:** Did your architectural modification causally improve success in a controlled ablation, or is that still a hypothesis?

### 25.3 Work through one RM65 example on paper

Task: `put the red block into the tray`.

At one training anchor:

```text
front_image[t]                [C,H,W]
wrist_image[t]                [C,H,W]
state[t]                      [D_s]
language                      "put the red block into the tray"
plan[t]                       "reach red block" (illustrative)
future_clean_actions[t:t+H]    [H,D_a]
action_valid_mask              [H,1]
box_valid_mask                 1 only on annotated keyframe
FAST_tokens                   tokenized chunk, padded + token mask
```

Then answer:

- What if the action chunk spans `reach → grasp` but the Plan stays `reach`?
- How much action supervision remains near that boundary?
- What happens if `red block` is incorrectly grounded but the velocity field is perfect?
- Can an offline loss detect the wrong object choice?

The point is to **connect Data → Architecture → Loss → Real Robot**.

---

# Part VII — Exercises, Review and Mock Interview

## 26. Hand Calculations

### Exercise A — Chunk mask

An episode has length `T=7`, action horizon `H=5`, and sample anchor `t=4`. Under the convention that action `a_t` is available at frame `t`:

1. List all valid actions in the chunk.
2. List the binary timestep mask.
3. If `D=6`, how many valid scalar action dimensions does the fully expanded mask contain?

### Exercise B — Effective batch

`world_size=3`, per-GPU microbatch `2`, gradient accumulation `6`. What is the global effective batch? How many optimizer updates occur per rank after 120 local microbatches, if no remainder is dropped?

### Exercise C — Masked loss

For a 2-step action chunk with \(D=2\), squared errors are:

\[
E=\begin{bmatrix}1&9\\25&49\end{bmatrix}
\]

Only the first action step is valid. Compute the valid-element mean squared error. Why is `.mean()` after zeroing invalid error entries not the same?

### Exercise D — Memory

Under the illustrative 16-byte/parameter accounting convention, how much *parameter-side* memory does a 1.2B model require? Why is that **not** a sufficient GPU requirement estimate?

---

## 27. Interview Questions — Basic (10)

1. What is the difference between frame, control step, episode, and action chunk?
2. Define `observation[t]`, `state[t]`, and `action[t]` for your RM65 data.
3. Why is camera/action synchronization necessary?
4. Why is nearest-neighbor image matching not always safe for causal training?
5. What is action horizon? How is it different from execution horizon?
6. What is the difference between joint position target and joint delta?
7. Why must normalization be fitted on train split only?
8. What do a padded action zero and a true zero velocity mean?
9. How does an action mask enter the Flow Matching loss?
10. What tensors are needed for a π0-style training batch?

## 28. Interview Questions — Intermediate (10)

11. Your training split is random across frames. What bias could this introduce?
12. Why can two `[B,50,7]` policies be incompatible despite matching tensor shapes?
13. What is an off-by-one action label error, and how would you detect it?
14. How do you supervise variable-length action chunks?
15. How does subtask-boundary truncation affect loss statistics?
16. What is the correct order of `backward`, clipping, optimizer and scheduler steps?
17. What is effective batch size with DDP and gradient accumulation?
18. How are BF16 and FP16 different in numerical behavior?
19. Why is a 16 bytes/parameter memory estimate not the whole story?
20. What does FSDP shard that DDP replicates?

## 29. Interview Questions — Advanced (10)

21. A 3B VLA OOMs during backward but not forward. Rank your first three interventions and justify.
22. You add more GPUs with DDP but each GPU still OOMs. Why?
23. Why might using `model.no_sync()` incorrectly silently harm distributed accumulation?
24. A multi-task VLA gets great loss on short clips but fails at Plan transitions. What would you inspect?
25. Plan loss improves, yet real-robot success drops. Name three causal hypotheses and design ablations.
26. Your camera is delayed by 150 ms relative to state/action; what happens and how do you fix it?
27. Under mixed action dimensions, how do you compute the denominator for masked MSE?
28. You load a checkpoint but actions move three times farther than intended. What do you audit?
29. A training run is 30% slower after enabling FSDP. Is that necessarily a bug?
30. For a fixed dataset of 442 episodes, how would you distinguish model underfitting, label noise, and overfitting?

---

## 30. Reference Solutions — Hand Calculations

**A.** Valid actions: `a4, a5, a6`; mask `[1,1,1,0,0]`. Valid scalar elements: `3×6=18`.

**B.** Global batch \(2\times3\times6=36\) episodes/windows per optimizer update (assuming conventional equal batches). `120/6=20` optimizer updates per rank.

**C.** Only `1` and `9` are valid: `(1+9)/2 = 5`. Naively zeroing invalid entries and taking the `[2,2]` mean gives `(1+9)/4=2.5`, incorrectly reducing the loss.

**D.** `1.2×10^9×16 = 19.2 GB` (decimal), **before** activations, temporary buffers, communication, and allocator overhead. Actual mixed-precision layouts can differ from this convention.

---

## 31. Final Self-Test Rubric

Score each item `0–3`:

- **0** cannot explain;
- **1** understand only when reading notes;
- **2** explain independently;
- **3** derive/implement and handle follow-up questions.

| Topic | Score (0–3) | Your missing details |
|---|---:|---|
| Timestamp alignment | | |
| Observation/action label convention | | |
| Joint/EEF & absolute/delta | | |
| Episode-aware chunking | | |
| Subtask-boundary truncation | | |
| Padding/DoF/attention masks | | |
| Split & normalization leakage | | |
| LeRobot dataset windows | | |
| Flow-matching training step | | |
| Loss normalization | | |
| Auxiliary Plan/Box/FAST losses | | |
| AdamW / grad accumulation | | |
| BF16 and clipping | | |
| Parameter vs activation memory | | |
| DDP / FSDP2 / ZeRO | | |
| OOM / NaN diagnosis | | |
| Dataset-to-real-robot generalization | | |
| RM65 interview walkthrough | | |

**Pass criterion:** Average `≥ 2.3`, all critical concepts (alignment, action semantics, mask, flow step, DDP/FSDP) `≥ 2`.

### 31.1 Final checklist

- [ ] I can define observation/action alignment without looking at code.
- [ ] I can hand-calculate an action window and its valid mask.
- [ ] I can explain why a zero-valued action is not necessarily padding.
- [ ] I can separate observation history from future action targets.
- [ ] I can write a masked Flow Matching loss from memory.
- [ ] I ran `week3_minimal_flow_train.py` and inspected at least one batch.
- [ ] I know what the checkpoint must store beyond weights.
- [ ] I can calculate effective global batch size.
- [ ] I can sketch DDP all-reduce and FSDP all-gather/reduce-scatter.
- [ ] I can explain ZeRO-1/2/3.
- [ ] I can debug OOM systematically rather than blindly lowering batch size.
- [ ] I can explain my RM65 training pipeline for 10 minutes.

---

## 32. One-Page Interview Cheat Sheet

```text
RAW DATA
  Define action semantics, units, coordinate frame and reference timestamps.
  Align cameras/state/actions on causal observation→command timeline.
  Audit dropped frames, bad demonstrations and stage labels.

DATASET
  Split by episode/scene before extracting overlapping chunks.
  Fit normalization only on training data.
  Sample current/past observations + future action targets.
  Pad at episode/subtask boundary and apply valid masks.

FLOW MATCHING STEP
  clean A                  [B,H,D]
  tau ~ Uniform(0,1)       [B,1,1]
  eps ~ Normal(0,I)        [B,H,D]
  x_tau = (1-tau)*eps + tau*A
  target = A-eps
  pred = action_expert(VLM_context, state, x_tau, tau)
  loss = sum(mask * (pred-target)^2) / sum(mask)

OPTIMIZATION
  zero_grad → forward → loss → backward → clip → optimizer → scheduler
  global batch = per-GPU microbatch × GPUs × accumulation steps
  BF16 autocast ≠ every tensor BF16
  checkpoint: weights + optimizer + step + action normalization + schema

DISTRIBUTED
  DDP: full model replicated, gradient all-reduce.
  ZeRO-1: optimizer sharded.
  ZeRO-2: optimizer + gradients sharded.
  ZeRO-3/FSDP full-shard: params + grads + optimizer sharded.

DEBUG
  OOM: identify load/forward/backward/step first.
  NaN: inspect raw data, normalizers, mask denominators, dtype and LR.
  Low loss, low real success: inspect semantics, grounding, control and OOD.
```

---

## 33. Optional Local Repository Layout

To stay consistent with your existing `assets/VLA_Interview/` directory:

```text
assets/VLA_Interview/
├── VLA_Interview_Week1_Action_Modeling_Foundations.md
├── VLA_Interview_Week2_Architecture_Evolution.md
├── VLA_Interview_Week3_Data_Training_Engineering.md  # this note
└── week3_minimal_flow_train.py                          # companion
```

You can also link them from your main `README.md` interview preparation section. The files delivered in chat are **local downloads**; they are not committed to GitHub automatically.

---

## 34. Primary References & Follow-up Reading

### Existing notes in your repo

- [FSDP](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Parallelism/FSDP.md)
- [Mixed Precision Training](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Training_Optimization/Mixed_Precision_Training.md)
- [Gradient Checkpointing](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Training_Optimization/Gradient_Checkpointing.md)
- [GPU Memory Estimation](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Math/Memory_Estimation.md)
- [AdamW](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Optimizer/AdamW.md)
- [Open X-Embodiment](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Model_Zoo/Robotics/Datasets/Open_X_Embodiment.md)
- [Diffusion Policy](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Model_Zoo/Robotics/Policies/Diffusion_Policy.md)
- [π0](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Model_Zoo/Robotics/Policies/Pi_0.md)
- [RDT-1B](https://github.com/XP-PY/llm-prep-2026/blob/main/docs/Model_Zoo/Robotics/Policies/RDT_1B.md)

### Official engineering documentation

- [LeRobot Dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobot Dataset API](https://huggingface.co/docs/lerobot/main/api/datasets)
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [DeepSpeed ZeRO](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/zero.md)

> **Next:** Week 4 — robot state/control foundations, VLA inference runtime, latency, action execution, real-robot evaluation, failure analysis, and Sim2Real.
