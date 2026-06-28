# [pi0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)

## Convenient Links

* [Paper (arXiv)](https://arxiv.org/abs/2410.24164)
* [Project Page / Blog](https://physicalintelligence.company/blog/pi0)
* [OpenVLA note in this repo](./OpenVLA.md)
* [Octo note in this repo](./Octo.md)
* [SmolVLA note in this repo](./SmolVLA.md)
* [Diffusion Policy note in this repo](./Diffusion_Policy.md)
* [ACT / ALOHA note in this repo](./ACT.md)

## 1. One-Sentence Summary

pi0 is a **3.3B-parameter generalist robot policy** from Physical Intelligence that starts from a pretrained **PaliGemma VLM**, adds a **300M-parameter action expert**, trains on over **10,000 hours of robot manipulation data** across many robot embodiments, and uses **conditional flow matching** to generate continuous high-frequency action chunks for dexterous real-world control.

## 2. Why pi0 Matters

Earlier VLA systems such as RT-2 and OpenVLA showed that pretrained vision-language models can be adapted for robot control, but they often represent actions as discrete tokens.

That is a clean interface for language models, but it is awkward for dexterous control:

* robot actions are continuous
* fine manipulation needs high precision
* bimanual manipulation often needs coordinated action chunks
* control may run at `20 Hz` to `50 Hz`
* long tasks require recovery from mistakes, not just one-step prediction

pi0 makes a different design choice:

> Keep the semantic strength of a pretrained VLM, but attach a continuous flow-matching action generator for robot control.

The paper matters because it combines three ingredients at a large real-robot scale:

* **VLM initialization** from PaliGemma, importing Internet-scale visual and language priors
* **cross-embodiment robot pretraining** across single-arm, bimanual, and mobile manipulators
* **post-training on high-quality task data**, similar in spirit to LLM alignment after broad pretraining

The result is not only a benchmark model. It is a recipe for a robot foundation model:

```text
large diverse robot data
-> pretrained VLM backbone
-> continuous action expert
-> broad base policy
-> task-specific post-training
-> dexterous long-horizon robot behavior
```

## 3. Core Idea

pi0 models a language-conditioned robot policy:

$$
\pi(A_t \mid o_t)
$$

where the observation is:

$$
o_t =
\left[
I_t^1,\,
\dots,\,
I_t^n,\,
\ell_t,\,
q_t
\right]
$$

and includes:

* `2` or `3` RGB camera images
* a language instruction
* proprioceptive robot state, such as joint angles

The output is an action chunk:

$$
A_t =
\left[
a_t,\,
a_{t+1},\,
\dots,\,
a_{t+H-1}
\right]
$$

The paper uses:

$$
H = 50
$$

This chunking is important. Instead of predicting one action at a time, pi0 predicts a short future trajectory, which makes high-frequency dexterous control more practical.

## 4. High-Level Pipeline

The full training and deployment story is:

```text
PaliGemma VLM initialization
-> add proprioceptive state input
-> add action expert for noisy action chunks
-> pretrain on broad cross-embodiment robot data
-> optionally post-train on curated task-specific data
-> sample continuous action chunks with flow matching
-> execute chunks open-loop with periodic replanning
```

The model can be used in three main ways:

1. **Out-of-box prompting**
   * run the pretrained base policy directly with a language command
2. **Post-training**
   * fine-tune on high-quality data for a target task
3. **High-level language guidance**
   * use a VLM or human to produce intermediate language commands for long tasks

This is closer to the training pattern used for LLMs than to the classic robotics pattern of training one policy from scratch for one task.

## 5. Architecture Overview

pi0 consists of two coupled parts:

```text
PaliGemma VLM backbone
-> image and language tokens

robotics action expert
-> proprioceptive state token
-> noisy action chunk tokens
-> denoising vector field
```

The parameter count is:

| Component          | Size              |
| :----------------- | :---------------- |
| PaliGemma backbone | about `3B` params |
| Action expert      | about `300M`      |
| Full pi0 model     | about `3.3B`      |

The action expert is initialized from scratch. The VLM backbone is initialized from PaliGemma, which gives the model a pretrained visual-language interface before it ever sees robot actions.

## 6. PaliGemma Backbone

The paper uses **PaliGemma** as the base VLM because it is small enough for practical control while still bringing strong vision-language pretraining.

The standard VLM input path handles:

```text
RGB images -> visual embeddings
language prompt -> language tokens
```

pi0 augments this with robot-specific inputs:

```text
proprioceptive state q_t
noisy action chunk A_t^tau
```

The important architectural choice is that these robot-specific tokens are not simply forced through the exact same weights as ordinary image and text tokens. They are routed through a smaller action expert.

## 7. Action Expert as a Two-Expert Transformer

pi0 is implemented as one transformer-style model with two sets of weights:

| Token type                  | Routed to                  |
| :-------------------------- | :------------------------- |
| Images and language prompt  | pretrained VLM backbone    |
| Robot state and action data | robotics action expert     |

The experts interact through attention, but the action expert has its own weights for robot-specific computation.

This matters because robot state and noisy action tokens are not part of PaliGemma's original pretraining distribution. Giving them a separate expert reduces the burden on the VLM backbone while still letting the robot policy condition on VLM features.

## 8. Attention Mask

Appendix B describes a blockwise causal attention layout with three blocks:

```text
[images, language] -> [state] -> [noisy action chunk]
```

Within each block, attention is bidirectional. Across blocks, earlier blocks do not attend to later blocks.

The practical reasons are:

* image and language tokens stay close to the original PaliGemma pretraining format
* state tokens can be cached during flow-matching sampling
* action tokens can attend to the full observation and to each other
* the action chunk can be generated coherently, rather than as isolated actions

Compared with autoregressive action-token prediction, this design is much more natural for continuous action chunks.

## 9. Flow-Matching Action Generation

pi0 does not discretize actions into vocabulary tokens.

Instead, it learns a conditional flow model over continuous action chunks.

Let the clean action chunk be:

$$
A_t =
\left[
a_t,\,
\dots,\,
a_{t+H-1}
\right]
$$

The noisy action chunk is sampled by interpolating between Gaussian noise and the real action chunk:

$$
A_t^\tau
=
\tau A_t
+
(1 - \tau)\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$

where:

$$
\tau \in [0, 1]
$$

The model predicts a vector field:

$$
v_\theta(A_t^\tau, o_t)
$$

The target vector field is:

$$
u(A_t^\tau \mid A_t)
=
A_t - \epsilon
$$

The flow-matching loss is:

$$
\mathcal{L}_\tau(\theta)
=
\mathbb{E}
\left[
\left\|
v_\theta(A_t^\tau, o_t)
-
u(A_t^\tau \mid A_t)
\right\|_2^2
\right]
$$

The interpretation is:

```text
start from noise
-> repeatedly predict a direction toward a realistic action chunk
-> integrate the vector field
-> get continuous robot actions
```

## 10. Inference with Euler Integration

At inference time, pi0 starts from random noise:

$$
A_t^0 \sim \mathcal{N}(0, I)
$$

and integrates the learned vector field:

$$
A_t^{\tau + \delta}
=
A_t^\tau
+
\delta v_\theta(A_t^\tau, o_t)
$$

The paper uses:

| Setting           | Value  |
| :---------------- | :----- |
| Flow steps        | `10`   |
| Step size         | `0.1`  |
| Action horizon    | `50`   |
| Typical frequency | up to `50 Hz` |

This is why the model can run as a real-time robot policy despite using an iterative generative process.

The paper reports inference timing on an NVIDIA RTX 4090:

| Model part                    | Time    |
| :---------------------------- | ------: |
| Image encoders                | `14 ms` |
| Observation forward pass      | `32 ms` |
| 10 action forward passes      | `27 ms` |
| Total on-board inference      | `73 ms` |
| Total off-board inference     | `86 ms` |

The model caches observation keys and values, so the repeated flow steps mainly recompute the action-token suffix.

## 11. Action Execution

Because pi0 predicts an action chunk, the robot does not need to run full inference at every control step.

The paper executes chunks open-loop and replans periodically:

| Robot family       | Control rate | Replanning cadence |
| :----------------- | :----------- | :----------------- |
| UR5e / Franka      | `20 Hz`      | every `0.8 s`, after `16` actions |
| Other robots       | `50 Hz`      | every `0.5 s`, after `25` actions |

The authors tried temporal ensembling early on, following ACT-style action chunking, but found that it hurt policy performance, so the final setup executes action chunks without aggregation.

## 12. Data Recipe

pi0's data recipe is one of the main contributions.

The pretraining mixture combines:

* Physical Intelligence's own dexterous robot data
* open-source datasets from Open X-Embodiment, Bridge v2, and DROID

The headline numbers are:

| Data / setup                         | Quantity |
| :----------------------------------- | -------: |
| Total robot data                     | over `10,000` hours |
| PI robot configurations              | `7` |
| PI tasks                             | `68` |
| PI robot timesteps                   | `903M` |
| Single-arm PI timesteps              | `106M` |
| Dual-arm PI timesteps                | `797M` |
| Open-source share of training mixture | `9.1%` |
| OXE robot coverage                   | `22` robots |

The paper emphasizes that "task" is broad here. For example, "bussing" is not just one pick-place pair. It includes putting many kinds of dishes, cups, utensils, and trash items into the correct receptacles.

## 13. Cross-Embodiment Training

The model is trained jointly on multiple robot types:

| Robot setup                         | Notes |
| :---------------------------------- | :---- |
| UR5e                                | single arm, wrist and over-the-shoulder cameras |
| Bimanual UR5e                       | two UR5e arms, three cameras |
| Franka                              | single Franka arm, two cameras |
| Bimanual Trossen                    | ALOHA-style two-arm setup |
| Bimanual ARX / AgileX               | two 6-DoF arms, wrist and base cameras |
| Mobile Trossen / Mobile ARX         | bimanual mobile manipulator |
| Mobile Fibocom                      | bimanual robot on a holonomic base |

To make one model work across these embodiments, the paper standardizes state and action tensors:

```text
state/action dimension = largest robot dimension
smaller robots -> zero padding
missing camera views -> masked image slots
```

The largest vector dimension is `18`, enough for two 6-DoF arms, two grippers, a mobile base, and a vertically actuated torso.

This is a simple but important engineering point: cross-embodiment learning requires a common tensor interface, even if each robot physically exposes a different action space.

## 14. Pretraining vs. Post-Training

The paper explicitly borrows the pretraining/post-training split from large language models.

### 14.1 Pretraining

The goal of pretraining is broad capability:

* many robots
* many scenes
* many object types
* many partial failures and recoveries
* many action distributions

The authors argue that diverse pretraining data teaches robustness. It exposes the model to states that polished expert data may not contain.

### 14.2 Post-Training

The goal of post-training is task fluency:

* consistent execution style
* efficient trajectories
* high-quality demonstrations
* specialization to complex downstream tasks

This mirrors LLM alignment:

```text
pretraining -> broad knowledge
post-training -> desired behavior
```

For robots, the "desired behavior" is not just politeness or instruction following. It is physical dexterity, reliability, recovery, and efficient task completion.

## 15. Language and High-Level Policies

Some tasks are too semantically complex to solve well from a single flat instruction.

For example:

```text
bus the table
```

may require many decisions:

```text
pick up the napkin
put the napkin in the trash
pick up the plate
shake trash off the plate
put the plate in the dish bin
...
```

Because pi0 is language-conditioned, it can receive these intermediate commands from:

* a human expert
* a high-level VLM policy
* a flat task prompt

This separates semantic task decomposition from low-level dexterous control:

```text
high-level VLM or human
-> intermediate language command
-> pi0 executes continuous robot actions
```

The paper reports that the PaliGemma-initialized pi0 follows intermediate language commands much better than the non-VLM pi0-small baseline.

## 16. Out-of-Box Evaluation

The first evaluation uses the base pretrained model without post-training.

Tasks include:

| Task                | Robot setup       | Skill tested |
| :------------------ | :---------------- | :----------- |
| Shirt folding       | bimanual ARX      | deformable object manipulation |
| Bussing easy        | UR5e              | object sorting and semantic recognition |
| Bussing hard        | UR5e              | clutter, occlusion, unseen objects |
| Grocery bagging     | UR5e              | multi-object packing |
| Toast out of toaster | bimanual Trossen | precise bimanual manipulation |

The baselines include:

* OpenVLA trained on the same mixture
* OpenVLA trained only on UR5e data
* Octo trained on the same mixture
* pi0-small without VLM initialization
* a compute-parity pi0 trained for `160k` steps instead of the full `700k`

The headline result is qualitative but strong:

* full pi0 performs best across the out-of-box tasks
* compute-parity pi0 still outperforms OpenVLA and Octo
* pi0-small beats OpenVLA and Octo but trails full pi0

The authors attribute OpenVLA's weakness in this setting largely to its autoregressive discrete action-token formulation, which does not naturally support high-frequency action chunks.

## 17. Language-Following Evaluation

The language evaluation compares:

| Condition       | Description |
| :-------------- | :---------- |
| `pi0-flat`      | receives only the high-level task command |
| `pi0-human`     | receives intermediate commands from a human |
| `pi0-HL`        | receives intermediate commands from a high-level VLM |
| `pi0-small-flat` / `pi0-small-human` | non-VLM baseline variants |

Tasks include:

* bussing
* table setting
* grocery bagging

The important result is that full pi0 benefits much more from intermediate language commands than pi0-small.

This supports the paper's claim that VLM pretraining matters not only for image recognition, but also for language-conditioned robot control.

## 18. Fine-Tuning to New Dexterous Tasks

The paper fine-tunes pi0 on tasks that differ from pretraining data:

| Task                    | Robot      | Difficulty framing |
| :---------------------- | :--------- | :----------------- |
| Stack bowls             | UR5e       | similar to pretraining |
| Towel folding           | bimanual ARX | similar to shirt folding |
| Tupperware in microwave | bimanual ARX | new appliance, related manipulation |
| Paper towel replacement | bimanual UR5e | hard, new objects and motions |
| Items in drawer         | Franka     | hard, limited similar pretraining data |

The comparisons include:

* pi0 fine-tuned from the pretrained base model
* pi0 trained from scratch on the task data
* OpenVLA
* Octo
* ACT
* Diffusion Policy

The main lesson is:

> pi0's architecture is already strong for dexterous control, and broad robot pretraining often improves it further, especially when fine-tuning data is limited or the task is close to the pretraining distribution.

The paper notes that on some tasks, prior methods trained from scratch are the strongest non-pi0 baselines. This is an important result because it shows that useful robot pretraining is still hard; simply having a pretrained robot model is not enough.

## 19. Complex Multi-Stage Tasks

The final evaluation studies long, difficult tasks:

| Task             | Main challenge |
| :--------------- | :------------- |
| Laundry folding  | deformable clothing from random crumpled states |
| Mobile laundry   | laundry folding with mobile base control |
| Dryer unloading  | navigation, opening dryer, transferring clothes |
| Table bussing    | clutter, semantic sorting, unseen objects |
| Box building     | deformable cardboard, bimanual bracing and folding |
| To-go box        | packing food and closing a flexible container |
| Packing eggs     | delicate grasping, placement, and carton closing |

These tasks combine:

* long horizons
* object diversity
* bimanual coordination
* physical deformation
* partial failure recovery
* semantic sequencing

The paper reports that full pretraining plus post-training performs best across these tasks, with the largest gains on the hardest settings.

A key claim is that these tasks go beyond the typical pick-and-place style benchmark and demonstrate a new level of learned dexterous manipulation.

## 20. Why Flow Matching Helps

Flow matching is a good fit for pi0 because robot actions are:

* continuous
* often multimodal
* temporally correlated
* high-frequency
* sensitive to small errors

Discrete action-token VLAs have to quantize each action dimension and predict those bins as text-like tokens. That can work well for coarse control, but dexterous tasks expose its limitations.

Flow matching instead models:

$$
p(A_t \mid o_t)
$$

directly in continuous action space.

This gives pi0 a more natural interface for:

* smooth motions
* bimanual coordination
* action chunks
* high-rate control
* precise gripper and end-effector behavior

In this sense, pi0 sits between two research lines:

```text
VLA models
-> pretrained visual-language semantics

diffusion / flow robot policies
-> continuous action generation
```

pi0's contribution is to combine them at robot-foundation-model scale.

## 21. Comparison to Nearby VLA Models

| Model      | Backbone idea | Action representation | Main emphasis |
| :--------- | :------------ | :-------------------- | :------------ |
| RT-2       | large VLM co-fine-tuned on robot and web data | discrete action tokens | semantic transfer from web-scale VLMs |
| OpenVLA    | open Prismatic VLM fine-tuned on Open X-Embodiment | discrete action tokens | open-source generalist VLA |
| Octo       | transformer over flexible robot tokens | diffusion action head | open generalist robot policy and adaptation |
| SmolVLA    | compact frozen VLM plus flow action expert | continuous flow action chunks | affordable, efficient VLA training |
| pi0        | PaliGemma plus action expert | continuous flow action chunks | dexterous cross-embodiment robot foundation model |

The key difference from OpenVLA and RT-2 is the continuous action generator.

The key difference from Octo and Diffusion Policy is the stronger pretrained VLM backbone and the scale of cross-embodiment dexterous robot data.

## 22. Limitations

The paper is explicit that pi0 is not a solved general robot brain.

Main limitations include:

* the best composition and weighting of pretraining data is still unclear
* not all evaluated tasks work reliably
* it is hard to predict how much data a new task will require
* the system is still focused on manipulation, not all embodied domains
* positive transfer across very different domains remains open
* complex tasks still often require post-training
* high-level planning may still need a separate VLM or human-specified intermediate commands

The authors specifically leave open whether this style of universal robot pretraining extends to domains such as autonomous driving, navigation, and legged locomotion.

## 23. Practical Takeaways

The main engineering lessons are:

1. **Continuous action heads matter**
   * dexterous control is a poor fit for pure text-token prediction

2. **Action chunks matter**
   * predicting a short trajectory is more practical than predicting one action at a time for high-frequency control

3. **VLM pretraining matters**
   * it improves language following and semantic grounding

4. **Pretraining and post-training solve different problems**
   * pretraining gives robustness and breadth
   * post-training gives fluent task execution

5. **Cross-embodiment learning needs boring interfaces**
   * padding action vectors and masking missing cameras are simple, but necessary

6. **Generalist policies still need data curation**
   * more robot data helps, but task balance and data quality are central

## 24. Mental Model

The shortest way to remember pi0 is:

```text
PaliGemma gives pi0 visual-language understanding.
The action expert gives pi0 continuous robot control.
Flow matching gives pi0 smooth high-frequency action chunks.
Pretraining gives pi0 broad physical experience.
Post-training gives pi0 task fluency.
```

So pi0 is not just "a VLM for robots."

It is better understood as:

> a robot foundation model recipe that combines pretrained VLM semantics, continuous flow-based action generation, cross-embodiment robot data, and task-specific post-training.
