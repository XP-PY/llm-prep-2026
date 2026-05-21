# [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)

## Convenient Links
* [Paper (arXiv)](https://arxiv.org/abs/2306.03310)
* [Project Page](https://libero-project.github.io/)
* [Official Code](https://github.com/Lifelong-Robot-Learning/LIBERO)
* [Official Documentation](https://lifelong-robot-learning.github.io/LIBERO/html/)
* [Open X-Embodiment note in this repo](./Open_X_Embodiment.md)
* [Diffusion Policy note in this repo](./Diffusion_Policy.md)
* [OpenVLA note in this repo](./OpenVLA.md)

## 1. One-Sentence Summary

LIBERO is a **simulation benchmark for lifelong robot learning** that procedurally generates language-conditioned manipulation tasks, releases **130 standardized tasks** across four task suites, provides **50 human-teleoperated demonstrations per task**, and uses the benchmark to study how robot policies transfer declarative knowledge, procedural knowledge, and mixtures of both across a task stream.

## 2. Why LIBERO Matters

Most VLA and robot policy papers evaluate whether a model can solve a fixed set of tasks. LIBERO asks a different question:

> What happens when a robot must keep learning new manipulation tasks over its lifetime?

That matters because robot learning is not only about recognizing objects or parsing language. It also involves **procedural knowledge**:

* how to reach
* how to grasp
* how to place
* how to open or close something
* how to chain multiple object interactions

The paper's core contribution is therefore not a new policy model. It is a benchmark that makes this transfer problem measurable:

```text
procedurally generated manipulation tasks
-> controlled shifts in objects, spatial layouts, and goals
-> human demonstration data
-> lifelong imitation learning protocol
-> success-rate metrics for transfer and forgetting
```

From a VLA perspective, LIBERO is useful because it exposes a weakness that pure vision-language benchmarks often hide: a model can know what the instruction means but still fail because it forgets or mislearns the **behavioral procedure** needed to complete it.

## 3. Problem Setup: Lifelong Robot Learning

The paper frames each manipulation task as a finite-horizon MDP:

$$
M = (S, A, T, H, \mu_0, R)
$$

LIBERO uses sparse rewards and replaces the reward with a goal predicate:

$$
g: S \rightarrow \{0, 1\}
$$

A lifelong learner sees a sequence of tasks:

$$
\{T^1, T^2, \dots, T^K\}
$$

Each task is defined by:

$$
T^k = (\mu_0^k, g^k)
$$

where:

* $\mu_0^k$ is the initial state distribution
* $g^k$ is the task goal predicate

The same policy must keep learning as the stream progresses:

$$
\pi(a_t \mid s_t; T^k)
$$

The important lifelong-learning constraint is:

> when learning task `k`, the agent no longer has full access to tasks `1 ... k-1`.

This is what makes the benchmark different from ordinary multitask imitation learning.

## 4. Lifelong Imitation Learning

Sparse-reward RL is expensive, so LIBERO studies a more practical setting: the user provides a small demonstration dataset for each task.

For task `k`, the dataset is:

$$
D^k = \{\tau_i^k\}_{i=1}^{N}
$$

where each trajectory is:

$$
\tau_i^k =
(o_0, a_0, o_1, a_1, \dots, o_l)
$$

The observation includes visual input plus robot joint and gripper information. Since the observation can be non-Markovian, the paper treats the state as observation history:

$$
s_t \equiv o_{\le t} = (o_0, o_1, \dots, o_t)
$$

Training uses behavior cloning:

$$
\mathcal{L}_{\text{BC}}
=
\sum_{t}
L\left(
\pi(o_{\le t}; T^k),
a_t
\right)
$$

In practice, LIBERO uses a GMM-style action head, so the policy models a multimodal distribution over continuous end-effector actions.

## 5. Benchmark Design

LIBERO is built around a procedural generation pipeline:

```text
Ego4D human activity annotations
-> behavior templates
-> language instructions
-> scene and object selection
-> PDDL initial-state specification
-> PDDL goal predicates
-> Robosuite simulation task
-> human teleoperation demonstrations
```

The paper emphasizes that the generation pipeline can in principle create infinitely many tasks. The released benchmark then fixes a standardized set of tasks so methods can be compared consistently.

## 6. Procedural Task Generation

LIBERO generates tasks in three steps.

### 6.1 Behavioral Templates and Instructions

The authors start from **Ego4D**, a large human activity dataset with language annotations.

They extract common activity descriptions and turn them into manipulation templates such as:

```text
Open ...
Put ... in ...
Place ... on ...
```

Then they instantiate these templates with simulator-compatible objects and scenes.

For example:

```text
Open the top drawer of the cabinet and put the bowl in it.
```

This gives LIBERO natural-language task descriptions while keeping the tasks executable inside simulation.

### 6.2 Initial State Distribution

Given an instruction, LIBERO selects a matching scene layout and object set.

The initial state distribution $\mu_0$ is specified in **PDDL**, including:

* object categories
* object placements
* initial object states
* scene layout

This is important because the benchmark can control what changes across tasks.

For example, it can keep the objects fixed but vary their spatial relationships, or keep the layout fixed but vary which object must be manipulated.

### 6.3 Goal Predicates

The task goal is also specified through PDDL-style predicates.

LIBERO uses unary predicates such as:

```text
Open(X)
TurnOff(X)
```

and binary predicates such as:

```text
On(A, B)
In(A, B)
```

The simulator terminates successfully when all goal predicates are true.

This is a clean benchmark design choice: language instructions, initial states, and success criteria are tied together through an explicit symbolic task specification.

## 7. Dataset and Task Suites

The released LIBERO benchmark has **130 language-conditioned manipulation tasks**:

| Suite | Tasks | Main Shift | Knowledge Type |
|:---|---:|:---|:---|
| LIBERO-Spatial | 10 | same objects and goal, different spatial relationships | declarative spatial knowledge |
| LIBERO-Object | 10 | same layout style, different target objects | declarative object knowledge |
| LIBERO-Goal | 10 | same objects and layout, different goals | procedural task knowledge |
| LIBERO-100 | 100 | diverse objects, layouts, backgrounds, and behaviors | entangled declarative + procedural knowledge |

The first three suites are often referred to as **LIBERO-X** in the paper. They are deliberately small and controlled so the benchmark can isolate different transfer factors.

LIBERO-100 is broader. The paper further splits it into:

| Split | Tasks | Role in the Paper |
|:---|---:|:---|
| LIBERO-90 | 90 short-horizon tasks | supervised pretraining source |
| LIBERO-Long | 10 long-horizon tasks | downstream lifelong-learning evaluation |

The data release includes:

* **50 demonstrations per task**
* collected by human experts
* using teleoperation with a **3Dconnexion SpaceMouse**
* inside **Robosuite**

So the released demonstration count for the 130 benchmark tasks is:

$$
130 \times 50 = 6500
$$

The paper is not trying to compete with Open X-Embodiment-scale robot data. LIBERO is much smaller, but more controlled: its value is in **diagnosing transfer and forgetting**, not in maximizing pretraining scale.

## 8. What Each Suite Tests

### 8.1 LIBERO-Spatial

All tasks ask the robot to place a bowl on a plate, with the same object set.

The challenge is that two visually identical bowls differ by location or spatial relationship.

So success depends on remembering and grounding instructions like:

```text
the bowl on the left
the bowl near another object
the bowl in a particular spatial relation
```

This suite tests transfer of **spatial declarative knowledge**.

### 8.2 LIBERO-Object

Each task asks the robot to pick and place a unique object.

The spatial layout is controlled, but the object identity changes.

This suite asks whether the learner can continually acquire and retain **object concepts** without confusing them as new tasks arrive.

### 8.3 LIBERO-Goal

The object set and spatial relationships remain fixed, but the task goal changes.

So the robot must learn different behaviors over the same scene, such as placing, opening, moving, or combining object interactions.

This suite is the cleanest probe of **procedural knowledge transfer**.

### 8.4 LIBERO-100 and LIBERO-Long

LIBERO-100 mixes the sources of variation:

* objects
* layouts
* backgrounds
* short-horizon skills
* long-horizon skills
* different object interactions

The paper uses **LIBERO-90** for pretraining and **LIBERO-Long** for downstream lifelong-learning evaluation.

This split is important because it directly tests whether ordinary supervised pretraining on many manipulation tasks helps later lifelong learning.

## 9. Policies and Algorithms Evaluated

LIBERO evaluates both policy architectures and lifelong-learning algorithms.

### 9.1 Policy Architectures

The paper implements three vision-language policy architectures:

| Architecture | Visual Backbone | Temporal Backbone | Language Use |
|:---|:---|:---|:---|
| ResNet-RNN | ResNet | LSTM | FiLM into visual features and LSTM input |
| ResNet-T | ResNet | Transformer decoder | language as a transformer token |
| ViT-T | Vision Transformer | Transformer decoder | language token in ViT and temporal transformer |

All three produce a latent representation at each decision step and use a **GMM action head** to model continuous manipulation actions.

The language instruction is encoded with pretrained **BERT** embeddings by default.

### 9.2 Lifelong-Learning Algorithms

The paper compares five training strategies:

| Method | Category | Role |
|:---|:---|:---|
| SEQ L | sequential finetuning | lower-bound-style baseline |
| MTL | multitask learning | upper-bound-style baseline |
| ER | rehearsal | replay data from previous tasks |
| EWC | regularization | penalize movement of important parameters |
| PackNet | dynamic architecture | prune and freeze task-specific subnetworks |

This design lets the paper compare not only "which model is better", but also how model architecture and lifelong-learning algorithm interact.

## 10. Evaluation Metrics

LIBERO reports all metrics in terms of **success rate**, not behavior-cloning loss.

That distinction matters because imitation-learning loss can improve while closed-loop rollout success gets worse. In robot control, small prediction errors can compound into task failure.

The three main metrics are:

| Metric | Direction | Meaning |
|:---|:---:|:---|
| FWT | higher is better | forward transfer: how quickly the agent learns a new task |
| NBT | lower is better | negative backward transfer: how much it forgets old tasks |
| AUC | higher is better | overall success-rate curve across learning |

At a high level:

```text
high FWT = learns new tasks fast
low NBT = forgets old tasks less
high AUC = better overall lifelong performance
```

This is the right metric mix for LIBERO because a lifelong learner can fail in two different ways:

* it may learn the new task slowly
* it may learn the new task but destroy performance on old tasks

## 11. Experimental Protocol

The experiment setup is fairly controlled:

* each task has **50 demonstration trajectories**
* each task is trained for **50 epochs**
* evaluation happens every **5 epochs**
* each evaluation uses **20 test rollouts**
* maximum rollout length is **600**
* optimizer: **Adam**
* batch size: **32**
* learning rate: cosine schedule from `1e-4` to `1e-5`
* seeds: `{100, 200, 300}`
* total reported combinations: **180 experiments**
* compute: single **Nvidia A100** or **A40**

The paper also matches the policy architectures by compute budget, reporting about **13.5G FLOPs** for each policy.

## 12. Main Experimental Results

### 12.1 Architecture Matters Almost as Much as Algorithm Choice

The first clear finding is that **ResNet-T** and **ViT-T** are usually much stronger than **ResNet-RNN**.

That suggests temporal Transformers are better than RNNs for this benchmark, especially when the policy must reuse information over a short observation history.

Representative AUC values from Table 1:

| Suite | ER + ResNet-RNN | ER + ResNet-T | ER + ViT-T |
|:---|---:|---:|---:|
| LIBERO-Long | 0.08 | 0.32 | 0.25 |
| LIBERO-Spatial | 0.29 | 0.56 | 0.50 |
| LIBERO-Object | 0.17 | 0.44 | 0.57 |
| LIBERO-Goal | 0.26 | 0.49 | 0.38 |

The exception is meaningful: **ViT-T is strongest on LIBERO-Object under ER**, which suggests that ViT-style visual processing helps when object variety is the main challenge.

With PackNet, ViT-T is also notably strong on LIBERO-Long:

| Suite | PackNet + ResNet-T AUC | PackNet + ViT-T AUC |
|:---|---:|---:|
| LIBERO-Long | 0.25 | 0.34 |
| LIBERO-Spatial | 0.63 | 0.59 |
| LIBERO-Object | 0.60 | 0.56 |
| LIBERO-Goal | 0.75 | 0.76 |

So the architecture lesson is not "ViT always wins" or "ResNet always wins". The better summary is:

* Transformer temporal backbones are consistently useful.
* ViT helps when visual object variation is high.
* ResNet-T can be stronger when the suite emphasizes spatial/procedural transfer under replay.
* Architecture and lifelong-learning algorithm interact.

### 12.2 Sequential Finetuning Learns New Tasks Fast, But Forgets

Table 2 fixes the architecture to **ResNet-T** and compares lifelong-learning algorithms.

The striking result is that **SEQ L has the best FWT on every suite**, but its NBT is very high.

In other words:

```text
sequential finetuning
-> fast adaptation to the current task
-> severe forgetting of previous tasks
```

This is why FWT alone is not enough. A method can look good at learning the next task while being bad at lifelong learning.

### 12.3 ER Is the Most Robust General Baseline

The AUC comparison from Table 2 is:

| Method | LIBERO-Long | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal |
|:---|---:|---:|---:|---:|
| SEQ L | 0.15 | 0.20 | 0.26 | 0.22 |
| ER | 0.32 | 0.56 | 0.44 | 0.49 |
| EWC | 0.02 | 0.06 | 0.16 | 0.06 |
| PackNet | 0.25 | 0.63 | 0.60 | 0.75 |
| MTL | 0.48 | 0.83 | 0.54 | 0.80 |

The paper's interpretation:

* **ER** is robust across all task suites.
* **PackNet** is strong on LIBERO-X, especially Spatial/Object/Goal.
* **PackNet** is weaker on LIBERO-Long because splitting capacity across tasks can hurt forward transfer.
* **EWC** performs poorly and can impede learning in LLDM.
* **MTL** is still the upper-bound-style reference because it can train on all tasks jointly.

This is a useful benchmark result because it says simple replay remains hard to beat when the task stream is heterogeneous.

### 12.4 Language Embeddings Do Not Yet Use Semantics Well

The paper compares BERT, CLIP, GPT-2, and Task-ID embeddings on LIBERO-Long using ResNet-T + ER.

| Embedding | Dimension | FWT | NBT | AUC |
|:---|---:|---:|---:|---:|
| BERT | 768 | 0.48 | 0.32 | 0.32 |
| CLIP | 512 | 0.52 | 0.34 | 0.35 |
| GPT-2 | 768 | 0.46 | 0.34 | 0.30 |
| Task-ID | 768 | 0.50 | 0.37 | 0.33 |

The important result is not the small numeric differences. The paper reports no statistically significant difference.

That means the language encoder is mostly serving as a task identifier, not as a rich semantic representation that improves transfer.

For VLA work, this is a warning: simply plugging in pretrained language embeddings does not guarantee grounded task understanding.

### 12.5 Task Ordering Matters

The paper evaluates ER and PackNet under five different task orderings.

The result is that the same algorithm can show noticeably different performance depending on the order in which tasks appear. The effect is especially significant for PackNet.

This matters because a deployed robot does not get to choose a perfect curriculum. A useful lifelong learner should be robust to different orderings, not only strong under one convenient stream.

### 12.6 Naive Supervised Pretraining Can Hurt

The pretraining study is one of the paper's most interesting negative results.

The authors pretrain policies on **LIBERO-90** using behavior cloning, then evaluate downstream lifelong learning on **LIBERO-Long**.

The result:

> basic supervised pretraining can hurt downstream lifelong-learning performance.

This does **not** mean pretraining is useless for robotics. It means naive BC pretraining is not automatically aligned with the later lifelong-learning objective.

For later VLA systems, this is an important distinction:

* large robot pretraining can help
* but the data mixture, objective, adaptation method, and evaluation protocol matter
* more pretraining is not automatically better if it creates brittle or task-irrelevant features

## 13. Why the Dataset Design Is the Main Contribution

LIBERO's main strength is that the benchmark controls what type of knowledge changes.

Many robot datasets mix everything together:

```text
new object + new layout + new goal + new background + new behavior
```

That makes failures hard to diagnose.

LIBERO separates the problem:

```text
LIBERO-Spatial -> did the robot transfer spatial relations?
LIBERO-Object  -> did the robot transfer object concepts?
LIBERO-Goal    -> did the robot transfer procedural behavior?
LIBERO-100     -> what happens when these factors are entangled?
```

This is why the paper is best read as a **benchmark design paper** rather than a model paper.

## 14. Relation to VLA Papers

LIBERO is not itself a VLA model. It is a benchmark and data-generation framework.

Compared with [Open X-Embodiment](./Open_X_Embodiment.md):

* Open X-Embodiment is a large real-world cross-embodiment data pool.
* LIBERO is a controlled simulated benchmark for lifelong transfer and forgetting.

Compared with [Diffusion Policy](./Diffusion_Policy.md):

* Diffusion Policy focuses on action-generation structure.
* LIBERO focuses on task-suite design and lifelong-learning evaluation.

Compared with [OpenVLA](./OpenVLA.md):

* OpenVLA asks how to train and adapt an open VLA.
* LIBERO asks how to measure controlled knowledge transfer in language-conditioned robot manipulation.

The papers are complementary: LIBERO gives later robot-policy and VLA work a way to test whether policies actually retain and transfer skills, not just whether they solve a static benchmark once.

## 15. Limitations

LIBERO is valuable, but it should be interpreted carefully.

1. **It is simulation-based**
   * Results do not automatically imply real-world robustness.
2. **The original benchmark assumes shared low-level task structure**
   * The formulation keeps state/action/transition/horizon shared across tasks.
3. **The controlled suites are intentionally small**
   * Ten tasks is enough to expose forgetting, but not enough to represent all lifelong robot learning.
4. **Language semantics are underused by the tested policies**
   * The embedding study suggests that task descriptions mostly behave like IDs.
5. **Pretraining results are about naive BC pretraining**
   * The negative result should not be read as evidence against all robot pretraining.

## 16. Key Takeaways

1. **LIBERO is a benchmark for lifelong decision-making, not just a manipulation dataset**
2. **It releases 130 language-conditioned simulation tasks and 50 expert demonstrations per task**
3. **The four suites separate spatial, object, goal, and mixed transfer**
4. **Temporal Transformers outperform RNN-style policy backbones in most settings**
5. **Replay is a strong robust baseline, while PackNet is strong but more capacity-sensitive**
6. **Success rate is more meaningful than behavior-cloning loss for robot policy evaluation**
7. **Task order and pretraining can change lifelong-learning outcomes substantially**
8. **For VLA research, LIBERO is useful because it probes whether models retain and transfer procedural behavior, not just language-conditioned recognition**
