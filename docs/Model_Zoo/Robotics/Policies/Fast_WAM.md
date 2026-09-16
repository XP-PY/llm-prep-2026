# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

> **Brief:** Fast-WAM learns actions together with future-video prediction, but generates only actions at inference. A structured attention mask makes action prediction independent of future-video tokens, allowing a pretrained video backbone to provide useful representations without expensive future-video denoising. The main finding is that video co-training matters more than test-time imagination in the evaluated settings.

**Reference:** Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao, Tsinghua University and Galaxea AI, 2026. This note follows the supplied **arXiv:2603.16666v2 PDF, dated March 23, 2026**.

## Convenient Links

* [Paper, arXiv v2](https://arxiv.org/abs/2603.16666v2)
* [Official project page](https://yuantianyuan01.github.io/FastWAM/)
* [Official code](https://github.com/yuantianyuan01/FastWAM)
* [Diffusion Policy note](./Diffusion_Policy.md)
* [pi0.5 note](./Pi_0_5.md)
* [LIBERO note](../Datasets/LIBERO.md)

## 1. What Question Does Fast-WAM Ask?

A **World Action Model (WAM)** combines visual world modeling with action prediction. Many WAMs generate future observations before, or jointly with, actions. Fast-WAM separates two possible sources of their performance:

| Component | When it happens | Potential benefit or cost |
|:--|:--|:--|
| Video co-training | Training | Predicting future observations encourages representations that capture motion and interactions |
| Explicit future imagination | Inference | Generated futures can guide actions, but iterative video denoising adds substantial latency |

Let $o$ be the current visual observation, $l$ the language instruction, $a_{1:H}$ an action chunk of horizon $H$, and $v_{1:T}$ future visual observations. An imagine-then-execute policy can be expressed as

$$
p(a_{1:H}\mid o,l)
=\int p(v_{1:T}\mid o,l)\,
p(a_{1:H}\mid o,l,v_{1:T})\,dv_{1:T}.
$$

Conceptually, it predicts a possible future and then produces actions conditioned on that future. Implementations may generate video and actions jointly rather than use two strictly sequential modules.

Fast-WAM instead uses

$$
p_\theta(a_{1:H}\mid o,l)
=p_\theta(a_{1:H}\mid z(o,l)),
$$

where $z(o,l)$ is a representation computed from the current context by the video backbone. It is **not a sampled future video**. The backbone learns with future-video supervision, but the deployed policy does not need to render or denoise a future to act.

This is a direct policy at inference, not a planner that searches over candidate future trajectories.

## 2. Architecture and Information Flow

![Fast-WAM video and action branches with training and inference attention masks](../../../../assets/Fast_WAM/architecture_and_masks.png)

*The current observation is the shared visual context. Future-video tokens and action tokens both access it, but actions cannot access future-video tokens. At inference, the future tokens disappear. Cropped from Figure 2, p. 5.*

### 2.1 Components

| Component | Role |
|:--|:--|
| Pretrained Wan2.2-5B video DiT | Initializes the video/world-modeling backbone; DiT means Diffusion Transformer |
| Pretrained video VAE | Encodes camera images into latent visual tokens |
| Built-in T5 text encoder | Encodes the instruction; all token groups receive language through cross-attention |
| Action expert DiT | Generates continuous action chunks; hidden dimension 1024, approximately 1B parameters |
| Shared-attention Mixture-of-Transformer (MoT) structure | Connects the video and action branches subject to the attention mask |

The reported total model size is approximately **6B parameters**. Images from multiple cameras are concatenated into one image before entering the VAE. MoT here describes modality-specific transformer branches connected by attention, not a learned top-$k$ router selecting interchangeable experts.

### 2.2 Three token groups

1. **Current observation:** clean latent tokens $f_0$ from the first/current frame. The symbol represents a group of tokens, not necessarily one token.
2. **Future video:** noisy latent tokens for future frames, used during training for video prediction.
3. **Actions:** noisy action tokens processed by the action expert.

The attention mask is the central design choice. In the following table, rows are query groups and columns are the key/value groups they can read:

| Query group | Current observation | Future video | Actions |
|:--|:--:|:--:|:--:|
| Current observation | Yes | No | No |
| Future video | Yes | Yes | No |
| Actions | Yes | No | Yes |

Attention within the future-video group and within the action group is bidirectional. Language cross-attention is separate from this table.

**Why must the current observation also be isolated?** Blocking only direct action-to-future attention would not suffice: if current-observation tokens could read future tokens, actions could retrieve that future information indirectly through the observation. The mask blocks both routes. It also keeps the observation representation independent of the noisy actions.

**How can video training help actions if the branches cannot read each other?** They share the visual backbone and current-observation representations. The video-prediction loss supplies gradients that shape these shared parameters; actions then use the resulting representations. Sharing learned parameters does not require access to a particular future sample during the forward pass.

Consequently, deleting future tokens at inference does not remove an input that the action branch was allowed to rely on during training. This is a trained architectural property, not an inference-only shortcut applied to an arbitrary WAM.

## 3. Joint Flow-Matching Training

Fast-WAM uses the same flow-matching formulation for actions and future-video latents. Let the clean target $y$ be either an action chunk $a_{1:H}$ or future-frame VAE latents $z_{1:T}$. Here $z_{1:T}$ denotes video targets, distinct from the conditioning representation $z(o,l)$ above.

Sample Gaussian noise $\epsilon\sim\mathcal N(0,I)$ and a noise level $t\in(0,1)$, then form

$$
y_t=(1-t)y+t\epsilon.
$$

At $t=0$ this is clean data; at $t=1$ it is pure noise. The target velocity follows directly by differentiation:

$$
\frac{dy_t}{dt}=\epsilon-y.
$$

The model predicts that velocity, with loss

$$
\mathcal L_{\mathrm{FM}}(y)
=\mathbb E_{y,\epsilon,t}
\left[\left\|f_\theta(y_t,t,o,l)-(\epsilon-y)\right\|_2^2\right].
$$

The two branch losses are combined:

$$
\mathcal L_{\mathrm{act}}=\mathcal L_{\mathrm{FM}}(a_{1:H}),
\qquad
\mathcal L_{\mathrm{vid}}=\mathcal L_{\mathrm{FM}}(z_{1:T}),
$$

$$
\boxed{\mathcal L=\mathcal L_{\mathrm{act}}+\lambda\mathcal L_{\mathrm{vid}}.}
$$

$\lambda$ balances action learning and video supervision; its numerical value is not specified in the PDF. The masks determine which inputs each branch of $f_\theta$ can access.

**Noise time is not robot time.** The variable $t$ indexes the flow from data to noise, while $H$ and $T$ index action and video horizons. At inference, action generation starts from noise and integrates the learned field in the reverse direction, from $t=1$ toward $t=0$.

## 4. Inference: What Is Removed, and What Remains?

```text
Current camera images + instruction
  -> VAE visual tokens + T5 language embeddings
  -> One video-backbone pass over clean current-observation tokens
  -> Reuse the resulting context for action generation
  -> Denoise an action chunk for 10 steps
  -> Output 32 actions
```

No future-video tokens are instantiated, so there is no iterative future-video generation. The **video backbone itself remains**: it encodes the current observation into the context used by the action expert.

The mask explains why the visual computation can be reused: observation tokens depend on neither future-video tokens nor the changing noisy action tokens. Their context does not need to be recomputed from those tokens at each action-denoising step.

**The entire policy is not a one-step action generator.** The paper's single-forward-pass description concerns obtaining the visual/world representation. Its implementation still uses **10 denoising steps**, with classifier-free guidance scale **1.0**.

The reported **190 ms** is inference latency on one NVIDIA RTX 5090D V2 32GB GPU. It is not a 190 ms duration for each physical action, and a 32-action chunk does not by itself specify the robot's controller frequency or how many actions execute before replanning.

## 5. Training Data and Settings

### 5.1 Pretraining versus downstream policy training

```text
Pretrained general video model: Wan2.2-5B
  -> Add the action expert
  -> Train on robot demonstrations with action + future-video objectives
  -> Evaluate the trained policy on the corresponding benchmark/task
```

**"Without embodied pretraining" does not mean training from scratch.** Fast-WAM starts from pretrained video-model weights. It avoids a separate large-scale robot-policy pretraining stage before the reported downstream training, but still learns from the robot demonstrations listed below. These results are not zero-shot robot control.

| Setting | Demonstrations | Training steps | Evaluation |
|:--|:--|:--|:--|
| LIBERO | Four suites, each with 500 demonstrations over 10 tasks; 2,000 demonstrations total | 20k | 2,000 trials across 40 tasks |
| RoboTwin 2.0 | 2,500 clean-scene and 25,000 randomized-scene demonstrations; multi-task training | 30k | 100 trials per task in each of the clean and randomized conditions |
| Real-world towel folding | 60 hours of teleoperation on Galaxea R1 Lite | 30k | Success rate and average task-completion time |

The RoboTwin setup is described as spanning more than 50 tasks; Appendix Table 3 gives the per-task breakdown. The real-world task is one deformable-object manipulation task, not a broad real-world task suite.

![Towel-folding sequence on the Galaxea R1 Lite robot](../../../../assets/Fast_WAM/towel_folding.png)

*The real-world evaluation requires coordinated manipulation of a deformable towel over multiple stages. Cropped from Figure 3, p. 7.*

| Hyperparameter | Reported setting |
|:--|:--|
| Action horizon | 32 actions |
| Video sampling | Temporal downsampling by 4; 9 video frames per chunk |
| Optimizer | AdamW |
| Learning rate / weight decay | $10^{-4}$ / 0.01 |
| Learning-rate schedule | Cosine annealing |
| Precision / gradient clipping | Mixed precision / 1.0 |
| Noise schedule | Described as logit-normal over $t$ for training and inference |
| Inference denoising / CFG | 10 steps / scale 1.0 |

The PDF does not provide a detailed freeze/unfreeze curriculum, batch size, or the complete noise-schedule parameterization. Those should not be inferred from the neighboring VLA recipes.

## 6. Controlled Variants: Separating the Two Effects

| Variant | Video co-training | Test-time future video | How actions are generated |
|:--|:--:|:--:|:--|
| **Fast-WAM** | Yes | No | Condition on current-observation representations |
| Fast-WAM-Joint | Yes | Yes | Jointly denoise video and actions, allowing attention between them |
| Fast-WAM-IDM | Yes | Yes | Generate future video first, then condition action prediction on it |
| Fast-WAM without video co-training | No | No | Same architecture and inference as Fast-WAM; remove the video objective |

**IDM** means inverse dynamics model: given the current observation and a desired/predicted future, infer actions connecting them. Its training uses noise augmentation on ground-truth video tokens with probability 0.5, following the compared design.

The authors align backbone, tokenization, and training recipes as closely as possible. Joint and IDM nevertheless change attention or conditioning, so they are controlled architectural comparisons, not merely the same checkpoint with video generation toggled off.

The no-video ablation still starts from the pretrained video backbone. It tests the value of **video co-training during robot-policy learning**, not whether general video pretraining is useful.

## 7. Results and Their Interpretation

### 7.1 RoboTwin 2.0

Success rates in percent, from paper Table 1. "Embodied PT" means the separate embodied pretraining reported for the baseline, not downstream training on RoboTwin demonstrations.

| Method | Embodied PT | Clean | Randomized | Average |
|:--|:--:|--:|--:|--:|
| pi0 | Yes | 65.92 | 58.40 | 62.2 |
| pi0.5 | Yes | 82.74 | 76.76 | 79.8 |
| Motus | Yes | 88.66 | 87.02 | 87.8 |
| Motus from Wan2.2 | No | 77.56 | 77.00 | 77.3 |
| LingBot-VA | Yes | 92.90 | 91.50 | 92.2 |
| LingBot-VA from Wan2.2 | No | 80.60 | Not reported | Not comparable* |
| **Fast-WAM** | No | **91.88** | **91.78** | **91.8** |
| Fast-WAM-Joint | No | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | No | 91.16 | 91.34 | 91.3 |
| Fast-WAM without video co-training | No | 82.76 | 84.80 | 83.8 |

*The source lists 80.6 in the average column for LingBot-VA from Wan2.2, but reports only its clean score. It is not a two-condition average like the other rows.*

Fast-WAM is close to pretrained LingBot-VA and slightly above both imagine-then-execute variants in the reported average. Removing video co-training costs **8.0 percentage points**, much more than the differences among the three video-co-trained variants.

### 7.2 LIBERO

Success rates in percent, from paper Table 2.

| Method | Embodied PT | Spatial | Object | Goal | Long | Average |
|:--|:--:|--:|--:|--:|--:|--:|
| OpenVLA | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| pi0 | Yes | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| pi0.5 | Yes | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | Yes | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | Yes | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **Fast-WAM** | No | **98.2** | **100.0** | **97.0** | **95.2** | **97.6** |
| Fast-WAM-Joint | No | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| Fast-WAM-IDM | No | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| Fast-WAM without video co-training | No | 89.2 | 99.2 | 95.4 | 90.0 | 93.5 |

Fast-WAM is competitive, but not the highest-scoring model. Removing video co-training costs **4.1 percentage points** overall, including **9.0** on Spatial and **5.2** on Long. By comparison, adding explicit future generation improves the reported average by only 0.9 points for Joint or 0.4 for IDM.

### 7.3 Real-world quality and inference speed

![Towel-folding success versus completion time and model inference latency](../../../../assets/Fast_WAM/real_world_results.png)

*Left: eventual task success versus physical task-completion time. Right: model inference latency, which is a different metric. Cropped from Figure 4, p. 9.*

The following success rates and completion times are approximate readings from the left-hand plot, not a separate numerical results table:

| Method | Success | Average completion time |
|:--|--:|--:|
| pi0.5 with pretraining | ~100% | ~120 s |
| **Fast-WAM** | **~75%** | **~150 s** |
| Fast-WAM-IDM | ~90% | ~180 s |
| Fast-WAM-Joint | ~70% | ~230 s |
| pi0.5 without pretraining | ~40% | ~205 s |
| Fast-WAM without video co-training | ~10% | ~240 s |

Pretrained pi0.5 remains the strongest real-world model in this comparison. Within the Fast-WAM family, IDM has higher success than Fast-WAM, while Fast-WAM completes the task faster. Thus, removing future generation produces a **speed/success tradeoff** here, not uniformly identical performance.

The latency values below are explicitly labeled in the right-hand plot:

| Method | Inference latency |
|:--|--:|
| pi0.5 | 180 ms |
| **Fast-WAM** | **190 ms** |
| Fast-WAM without video co-training | 190 ms |
| Fast-WAM-Joint | 580 ms |
| Fast-WAM-IDM | 810 ms |

Fast-WAM is approximately **3.1 times faster than Joint** and **4.3 times faster than IDM**. The paper's "over 4x" headline applies to the IDM comparison, not both variants. Video co-training improves the deployed policy without changing its reported inference latency relative to the no-video ablation.

## 8. Takeaways and Limits

1. **Separate learning a world representation from generating a future.** Predicting future video can be valuable supervision even when deployment requires only actions.
2. **The attention mask makes the separation possible.** Actions never rely on future-video tokens, directly or indirectly through observation tokens, during training.
3. **Single-pass visual encoding does not mean single-step action generation.** Fast-WAM retains iterative action denoising while removing iterative video denoising.
4. **The strongest evidence is the controlled ablation.** Dropping video co-training hurts more than dropping test-time imagination on the reported simulation averages; real-world results also show a substantial co-training benefit.
5. **Do not generalize this to all planning problems.** The real-world evaluation covers one task, IDM retains a success advantage there, and the PDF does not report uncertainty estimates or the real-world trial count. It also does not fully specify how failed trials enter the completion-time average. These results do not establish that explicit future prediction is unnecessary for every task or distribution shift.

**In one sentence:** Fast-WAM uses future-video prediction to improve a policy that acts from current observations, without requiring it to generate future video every time it acts.
