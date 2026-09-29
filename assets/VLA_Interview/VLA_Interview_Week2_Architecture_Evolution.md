# Week 2 — VLA Architecture Evolution & Model Comparison

> Goal: turn scattered paper knowledge into a **single architecture map** that can be recalled and explained in interviews.
>
> Main line:
>
> **ACT / Diffusion Policy → RT-1 → RT-2 → Octo → OpenVLA → π0 → FAST → π0.5 → SmolVLA → RDT → GR00T → WALL-OSS**
>
> For every model, always answer the same six questions:
>
> 1. **Input?**
> 2. **Backbone?**
> 3. **Action representation?**
> 4. **Training objective?**
> 5. **Training data / training recipe?**
> 6. **Inference procedure?**
>
> Suggested workload: **15–18 hours**
>
> Updated for interview preparation on **2026-09-28**.  
> In particular, the GR00T section uses the current **GR00T N1.7** stack instead of stopping at N1/N1.5.

---

# 0. What You Should Be Able to Do After This Week

By the end of Week 2, you should be able to answer without notes:

- Why ACT and Diffusion Policy are important even though they are not modern VLM-based VLAs.
- What architectural problem RT-1 solved.
- What conceptual jump RT-2 made from RT-1.
- Why Octo is important for cross-embodiment robot pretraining.
- Why OpenVLA became an important open VLA baseline.
- Why π0 moved back from discrete action tokens to continuous actions.
- What FAST solves and why FAST itself is **not** a VLA model.
- What π0.5 changed compared with π0.
- Why SmolVLA matters for affordable training/deployment.
- Why RDT is different from π0 even though both use generative continuous actions.
- What current GR00T N1.7 looks like architecturally.
- What WALL-OSS is trying to improve beyond “VLM + action head”.
- Given a new VLA paper, classify it by:
  - semantic backbone
  - modality fusion
  - action representation
  - action decoder
  - training stages
  - data scale
  - inference style
- Compare at least six models on a whiteboard in under 10 minutes.

---

# 1. Week Schedule

## Day 1 — Precursors: ACT + Diffusion Policy
**Estimated time: 2–2.5 h**

Review only the concepts that later VLAs inherit:

- action chunking
- temporal consistency
- multimodal action modeling
- CVAE vs diffusion
- receding-horizon control
- visual conditioning

Output:

```text
ACT:
chunking + temporal ensemble

Diffusion Policy:
continuous generative action chunks
```

---

## Day 2 — From Robot Transformer to VLA: RT-1 + RT-2
**Estimated time: 3 h**

Focus:

- RT-1: scalable language-conditioned robot policy
- RT-2: pretrained VLM + robot actions as tokens
- web knowledge transfer
- robot/web co-fine-tuning
- action discretization
- latency implications

Output:

> Why is RT-2 not simply “a larger RT-1”?

---

## Day 3 — Generalist/Open Robot Policies: Octo + OpenVLA
**Estimated time: 3 h**

Focus:

- Open X-Embodiment
- cross-embodiment pretraining
- flexible observations/tasks/actions
- diffusion head vs tokenized actions
- robot pretraining vs VLM pretraining

Output:

> Octo and OpenVLA are both generalist policies trained on large robot mixtures. Why are their architectures fundamentally different?

---

## Day 4 — Modern Flow VLA: π0 + FAST + π0.5
**Estimated time: 3–3.5 h**

Focus:

- PaliGemma initialization
- action expert
- Flow Matching
- continuous chunks
- FAST tokenization
- heterogeneous pretraining
- semantic subtask hierarchy

Output:

```text
π0
vs
π0-FAST
vs
π0.5
```

---

## Day 5 — Current Practical Architectures
**Estimated time: 3–3.5 h**

Study:

- SmolVLA
- RDT
- GR00T N1.7
- WALL-OSS

Focus on:

- efficiency
- cross-embodiment action spaces
- frozen vs trainable VLM
- DiT / action expert
- human video data
- embodied VQA / reasoning supervision

---

## Day 6 — Architecture Matrix + Mock Interview
**Estimated time: 2–3 h**

Tasks:

- draw the full evolution map from memory
- fill the model matrix from memory
- answer the interview questions
- connect each architecture choice to your RM65 project

---

# 2. First: Not Everything Here Is Strictly a VLA

| Method | Strict modern VLA? | Main role in the evolution |
|---|---:|---|
| ACT | No | action chunking, imitation learning |
| Diffusion Policy | No | continuous generative action policy |
| RT-1 | Borderline / precursor | large-scale language-conditioned robot Transformer |
| RT-2 | Yes | VLM knowledge transferred to actions |
| Octo | Generalist robot policy | cross-embodiment pretrained policy |
| OpenVLA | Yes | open VLM-based autoregressive VLA |
| π0 | Yes | VLM + continuous flow action expert |
| FAST | **No: tokenizer** | efficient discrete action representation |
| π0.5 | Yes | hierarchical + heterogeneous open-world VLA |
| SmolVLA | Yes | compact efficient flow VLA |
| RDT | Robot foundation policy | large diffusion Transformer for robot actions |
| GR00T | Yes | generalist humanoid/cross-embodiment VLA |
| WALL-OSS | Yes | tightly coupled embodied VLM + action model |

### Important interview point

Do not say:

> “FAST is a VLA model.”

FAST is primarily an **action tokenizer**.

`π0-FAST` is a model using FAST.

---

# 3. The Whole Evolution in One Picture

```text
                         ACTION MODELING
                              │
             ┌────────────────┴────────────────┐
             │                                 │
          ACT (2023)                    Diffusion Policy
             │                                 │
        action chunks                   generative chunks
             │                                 │
             └──────────────┬──────────────────┘
                            │
                            ▼
                     General robot policy
                            │
                  RT-1: robot Transformer
                            │
                            ▼
                   RT-2: VLM → actions
                            │
             ┌──────────────┴───────────────┐
             │                              │
             ▼                              ▼
          Octo                          OpenVLA
 cross-embodiment policy          open VLM + action tokens
 + diffusion head                       │
             │                           │
             └──────────────┬────────────┘
                            │
                            ▼
                           π0
             VLM + continuous action expert
                    + Flow Matching
                            │
                ┌───────────┴──────────┐
                │                      │
              FAST                   π0.5
 efficient tokenization       heterogeneous pretraining
                                 + semantic subtasks
                │                      │
                └───────────┬──────────┘
                            │
            ┌───────────────┼─────────────────────┐
            ▼               ▼                     ▼
         SmolVLA           RDT                 GR00T
      small flow VLA   large diffusion     generalist humanoid
                       robot foundation       VLA + DiT
                            │
                            ▼
                        WALL-OSS
              embodied VLM + semantic reasoning
                   + continuous control
```

This is a **conceptual evolution map**, not a strict family tree.

---

# 4. A Better Way to Read Any VLA Paper

For every model, build this template:

```text
Input
  ├─ image?
  ├─ language?
  ├─ robot state?
  └─ history?

Backbone
  ├─ vision encoder?
  ├─ language/VLM?
  └─ separate action expert?

Fusion
  ├─ concatenation?
  ├─ shared self-attention?
  ├─ cross-attention?
  └─ separate experts?

Action representation
  ├─ continuous?
  ├─ discrete bins?
  ├─ FAST?
  └─ joint / EEF / relative?

Training
  ├─ BC?
  ├─ CE?
  ├─ diffusion?
  ├─ flow matching?
  └─ auxiliary semantic losses?

Inference
  ├─ one forward?
  ├─ autoregressive decoding?
  ├─ iterative denoising?
  ├─ flow integration?
  └─ chunk execution / replanning?
```

This is more useful than memorizing paper section names.

---

# 5. ACT — The Action Chunking Precursor

## Six-Question Summary

### 1. Input?

- multi-view RGB
- current joint positions

No language input in the original ACT setup.

### 2. Backbone?

- ResNet18 image encoder
- Transformer encoder/decoder
- CVAE latent variable during training

### 3. Action Representation?

Continuous future joint-position chunk:

\[
A_t=[a_t,\dots,a_{t+k-1}]
\]

### 4. Training Objective?

\[
\mathcal L
=
\|A-\hat A\|_1
+
\beta D_{\mathrm{KL}}
\]

### 5. Training Data?

Small task-specific teleoperation datasets.

### 6. Inference?

- set latent \(z=0\)
- predict an action chunk
- query every timestep
- combine overlapping predictions with temporal ensembling

## What Later VLAs Inherit

\[
\boxed{\text{predict a trajectory chunk, not a single action}}
\]

This improves temporal consistency and reduces effective decision horizon.

---

# 6. Diffusion Policy — Continuous Generative Action Modeling

## Six-Question Summary

### 1. Input?

- image/state observation history

### 2. Backbone?

- visual encoder
- conditional action denoiser
- CNN or Transformer variants

### 3. Action Representation?

Continuous action sequence.

### 4. Objective?

\[
\mathcal L
=
\mathbb E
[
\|\epsilon-\epsilon_\theta(O,A^k,k)\|^2
]
\]

### 5. Data?

Task-specific demonstrations.

### 6. Inference?

```text
Gaussian action noise
→ repeated denoising
→ action chunk
→ execute first H_e actions
→ replan
```

## Historical Importance

Diffusion Policy demonstrated that:

\[
p(A_t\mid O_t)
\]

can be modeled as a rich multimodal distribution.

This directly influenced:

- diffusion action heads
- Flow Matching
- DiT action experts
- generative continuous VLA control

---

# 7. RT-1 — Scaling a Language-Conditioned Robot Transformer

## Core Question

> Can one Transformer policy absorb a large real-robot multi-task dataset and still run in real time?

## Six-Question Summary

### 1. Input?

- language instruction
- 6-frame RGB history

### 2. Backbone?

```text
instruction
→ Universal Sentence Encoder

images
→ FiLM-conditioned EfficientNet-B3
→ TokenLearner

tokens
→ decoder-only Transformer
```

Approximately 35M parameters.

### 3. Action Representation?

11 action dimensions.

Continuous dimensions are discretized into:

\[
256
\]

bins.

### 4. Training Objective?

Behavior cloning using categorical cross-entropy for discretized actions.

### 5. Training Data?

Approximately:

- 130k demonstrations
- 13 robots
- hundreds of instructions/tasks

### 6. Inference?

Closed-loop action prediction at robot control frequency.

The final design avoids expensive autoregressive action decoding.

## Key Contributions

- data scaling
- language-conditioned generalist control
- latency-aware token compression

TokenLearner compresses:

\[
81\rightarrow8
\]

visual tokens per frame.

---

# 8. RT-2 — The VLA Concept Becomes Explicit

RT-2 makes the key conceptual leap:

\[
\boxed{\text{robot action = another token sequence}}
\]

## Six-Question Summary

### 1. Input?

- image
- natural-language instruction

### 2. Backbone?

Large pretrained VLMs such as:

- PaLI-X
- PaLM-E

### 3. Action Representation?

Robot actions are discretized and mapped into vocabulary tokens:

\[
a^{(m)}
\rightarrow
\text{one of 256 bins}
\rightarrow
\text{token}
\]

### 4. Training Objective?

Autoregressive next-token cross-entropy:

\[
\mathcal L
=
-\sum_k
\log
p_\theta(y_k\mid y_{<k},I,\ell)
\]

### 5. Training Data?

Co-fine-tuning mixture:

```text
web vision-language data
+
robot trajectories
```

### 6. Inference?

```text
image + instruction
→ VLM
→ autoregressive action tokens
→ detokenize
→ robot action
```

---

# 9. RT-1 vs RT-2

| | RT-1 | RT-2 |
|---|---|---|
| Pretrained VLM knowledge | limited | central |
| Vision-language backbone | task architecture | large web-pretrained VLM |
| Action representation | discretized outputs | text-like vocabulary tokens |
| Web co-training | no | yes |
| Main goal | scale robot policy | transfer web semantics into robot control |

### Interview answer

> RT-1 shows that robot behavior cloning scales. RT-2 shows that pretrained web-scale visual-language knowledge can be transferred into robot actions by expressing actions in the same token space as language.

---

# 10. Why Action Tokens Were Attractive

A pretrained VLM already models:

\[
p(y_k\mid y_{<k},x)
\]

If actions become tokens, the robot problem can reuse:

- model architecture
- autoregressive decoder
- cross-entropy loss
- web co-training pipeline

But this introduces:

- quantization error
- long token sequences
- autoregressive latency

These weaknesses motivate later continuous-action VLAs.

---

# 11. Octo — Cross-Embodiment Robot Pretraining

Octo focuses on:

\[
\boxed{\text{one reusable robot policy across datasets and embodiments}}
\]

## Six-Question Summary

### 1. Input?

Flexible:

- language task
- goal image
- RGB observations
- multiple observation modalities

### 2. Backbone?

Transformer over:

- task tokens
- observation tokens
- readout tokens

Language is encoded with pretrained T5.

### 3. Action Representation?

Continuous action chunks.

### 4. Training Objective?

Diffusion action head.

### 5. Training Data?

Approximately 800k trajectories from Open X-Embodiment mixtures.

### 6. Inference?

```text
task tokens + observation tokens
→ transformer
→ readout token
→ diffusion action head
→ continuous action chunk
```

## Why Octo Matters

Architecture is modular:

```text
input tokenizer
    ↓
shared transformer
    ↓
readout
    ↓
action head
```

This makes adaptation easier for:

- new cameras
- new sensors
- new action spaces
- new embodiments

---

# 12. OpenVLA — An Open VLM-Based VLA

OpenVLA combines:

```text
large pretrained VLM
+
large robot dataset
+
discrete action token prediction
```

## Six-Question Summary

### 1. Input?

- image
- language instruction

### 2. Backbone?

Prismatic VLM.

Important visual ingredients include:

- SigLIP
- DINOv2

### 3. Action Representation?

7D robot action.

Each continuous dimension:

\[
a_i
\rightarrow
256\text{ bins}
\rightarrow
\text{token}
\]

### 4. Objective?

Autoregressive cross-entropy over action tokens.

### 5. Training Data?

Approximately:

\[
970k
\]

robot episodes from Open X-Embodiment.

### 6. Inference?

```text
image + instruction
→ VLM
→ action tokens
→ dequantization
→ continuous robot action
```

## Why It Matters

OpenVLA became a practical open baseline for:

- VLA pretraining
- LoRA fine-tuning
- FSDP
- Hugging Face integration
- robot deployment

A useful takeaway from its training study:

> adapting the visual encoder can matter for precise robot control.

---

# 13. Octo vs OpenVLA

| | Octo | OpenVLA |
|---|---|---|
| Core identity | generalist robot policy | VLM-based VLA |
| Main pretrained knowledge | robot mixture | web VLM + robot mixture |
| Action output | diffusion continuous chunk | discrete action tokens |
| Language | one task modality | central VLM interface |
| Architecture | modular tokenizers + transformer | pretrained VLM |
| Strength | cross-embodiment adaptation | semantic/language grounding |

---

# 14. π0 — Continuous Actions Return

π0 asks:

> Can we keep a pretrained VLM's semantic strength but avoid discretizing high-frequency robot actions?

Answer:

```text
VLM
+
continuous action expert
+
Flow Matching
```

## Six-Question Summary

### 1. Input?

- multiple RGB images
- language instruction
- proprioceptive robot state

### 2. Backbone?

Approximately:

```text
PaliGemma VLM ~3B
+
robot action expert ~300M
```

Total:

\[
\approx3.3B
\]

### 3. Action Representation?

Continuous action chunk:

\[
A_t=[a_t,\dots,a_{t+49}]
\]

with:

\[
H=50
\]

### 4. Training Objective?

Conditional Flow Matching:

\[
A^\tau=\tau A+(1-\tau)\epsilon
\]

\[
v^*=A-\epsilon
\]

\[
\mathcal L=\|v_\theta-v^*\|^2
\]

### 5. Training Data?

Over:

\[
10,000\text{ hours}
\]

of heterogeneous robot manipulation data.

### 6. Inference?

```text
encode image/language/state
→ initialize Gaussian action noise
→ predict velocity repeatedly
→ integrate flow
→ continuous action chunk
→ execute part of chunk
→ replan
```

---

# 15. The Key Architectural Idea in π0

π0 does not simply append action tokens to PaliGemma.

It introduces a robot-specific action expert:

```text
image + language
    ↓
pretrained VLM expert
    ↕ attention
state + noisy actions
    ↓
robot action expert
```

Why?

Robot-state/action representations are very different from text.

Separate weights let the model:

- preserve pretrained semantics
- specialize for control

---

# 16. OpenVLA vs π0

| | OpenVLA | π0 |
|---|---|---|
| Semantic base | pretrained VLM | pretrained VLM |
| Action | discrete tokens | continuous chunk |
| Generation | autoregressive | Flow Matching |
| Precision | quantized | continuous |
| Chunk generation | token-by-token | whole chunk updated in parallel |
| Action expert | no continuous expert | yes |

### Strong answer

> OpenVLA unifies language and actions by discretizing actions into tokens. π0 instead keeps actions continuous and attaches a Flow Matching action expert to a pretrained VLM, which is better suited to high-frequency dexterous action chunks.

---

# 17. FAST — Fixing Autoregressive Action Tokenization

FAST is not primarily a policy backbone.

It solves:

\[
\boxed{\text{how should continuous robot trajectories be turned into tokens?}}
\]

## Pipeline

```text
continuous trajectory
→ quantile normalization
→ DCT over time
→ quantization
→ frequency coefficients
→ BPE
→ compact token sequence
```

Why DCT?

Robot trajectories are smooth and low-frequency structure often captures much of the motion.

Why BPE?

Repeated coefficient patterns can be merged.

Result:

\[
n_{\text{tokens}}\ll H\times D
\]

for many trajectories.

---

# 18. π0 vs π0-FAST

### π0

```text
continuous actions
→ Flow Matching
```

### π0-FAST

```text
continuous actions
→ FAST tokenizer
→ autoregressive tokens
```

The main difference is the **action representation and training objective**.

---

# 19. π0.5 — Open-World Hierarchical VLA

π0.5 is not just π0 with more data.

Major changes:

1. heterogeneous pretraining
2. FAST action-token pretraining
3. continuous flow action expert for post-training/runtime
4. semantic subtask prediction
5. open-world mobile manipulation

## Six-Question Summary

### 1. Input?

- images
- overall task
- robot state
- semantic/task annotations during training

### 2. Backbone?

Retains the two-expert family:

```text
SigLIP + Gemma/PaliGemma-style VLM
+
~300M action expert
```

### 3. Action Representation?

Two representations across stages:

- FAST discrete tokens
- continuous Flow Matching actions

### 4. Training Objective?

Conceptually:

\[
\mathcal L
=
\mathcal L_{\mathrm{token}}
+
\alpha
\mathcal L_{\mathrm{flow}}
\]

### 5. Training Data?

Heterogeneous mixture:

- target mobile robot data
- other robot embodiments
- semantic subtask labels
- web/VLM data
- localization/VQA supervision

### 6. Inference?

Hierarchical:

```text
overall task
→ predict semantic subtask
→ condition low-level policy
→ Flow Matching action chunk
→ execute
→ update observation
→ next subtask
```

---

# 20. π0 vs π0.5

| | π0 | π0.5 |
|---|---|---|
| Main goal | general robot control | open-world long-horizon generalization |
| Action during main training | continuous flow | FAST + flow across stages |
| Explicit semantic hierarchy | limited/external | yes |
| Data recipe | broad robot pretraining | more heterogeneous robot + semantic + web |
| Runtime | task → actions | task → subtask → actions |

Key idea:

\[
\boxed{\text{semantic planning + low-level continuous control}}
\]

---

# 21. SmolVLA — Efficiency as the Main Design Goal

SmolVLA asks:

> Can modern VLA architecture work on small labs and affordable robots?

## Six-Question Summary

### 1. Input?

- one or more RGB images
- language
- robot state

### 2. Backbone?

Frozen SmolVLM-2.

Main scale:

\[
\approx450M
\]

### 3. Action Representation?

Continuous action chunks.

### 4. Training Objective?

Flow Matching.

### 5. Training Data?

Community LeRobot datasets from affordable robot platforms.

### 6. Inference?

Flow-generated chunks.

It also supports asynchronous inference:

```text
policy inference thread
        │
        ▼
action queue
        │
        ▼
robot execution thread
```

## Relative to π0

Main changes:

- much smaller VLM
- frozen VLM
- smaller data regime
- visual token reduction
- layer skipping
- affordable deployment
- asynchronous runtime

---

# 22. RDT — Robotics Diffusion Transformer

RDT is important because many robot-learning JDs mention it directly.

Official RDT-1B:

- roughly 1.2B parameters in the paper
- 1M+ multi-robot episodes
- up to 3 RGB views
- predicts a 64-step action sequence
- large Diffusion Transformer

## Six-Question Summary

### 1. Input?

- language instruction
- up to 3 RGB views
- robot state

### 2. Backbone?

```text
language
→ T5-v1.1-XXL

vision
→ SigLIP

state / actions
→ robot projections

all conditioning
→ Robotics Diffusion Transformer
```

### 3. Action Representation?

Continuous action chunk.

RDT introduces a **Physically Interpretable Unified Action Space**.

### 4. Training Objective?

Conditional diffusion / denoising objective over continuous action trajectories.

### 5. Training Data?

Pretraining:

\[
1M+
\]

multi-robot episodes.

Then:

\[
6k+
\]

self-collected bimanual episodes for fine-tuning.

### 6. Inference?

```text
language + images + state
→ condition DiT
→ start from noisy trajectory
→ iterative denoising
→ 64-step continuous action chunk
```

---

# 23. RDT's Unified Action Space

Robots differ in:

- joint count
- joint vs EEF control
- position vs velocity
- one vs two arms
- mobile base

RDT defines a large structured state/action vector with physically meaningful slots.

Conceptually:

```text
right-arm joint slots
left-arm joint slots
EEF translation
EEF rotation
gripper
base motion
...
```

Unused slots are masked/empty.

The official implementation also uses 6D rotation representation for EEF orientation.

### Why this matters

Cross-embodiment learning is not only:

> pad everything to the same length.

You also need semantic alignment:

> the same dimension should represent the same physical concept.

---

# 24. RDT vs π0

| | RDT | π0 |
|---|---|---|
| Core action method | diffusion | Flow Matching |
| Semantic backbone | T5 + SigLIP encoders | pretrained PaliGemma VLM |
| Main model | large DiT | VLM + action expert |
| Action horizon | 64 | 50 |
| Cross-embodiment | physically interpretable unified space | common padded interface |
| Identity | diffusion robot foundation model | VLA flow model |

---

# 25. GR00T — Current Generalist Humanoid VLA

As of 2026-09-28, NVIDIA's current open stack is:

\[
\boxed{\text{GR00T N1.7}}
\]

The architecture family is:

```text
VLM
+
state/action Diffusion Transformer
+
continuous generative actions
```

## Six-Question Summary

### 1. Input?

- language
- RGB images
- robot state

### 2. Backbone?

Current N1.7:

```text
Cosmos-Reason2-2B
(Qwen3-VL architecture)
+
Flow-Matching DiT action head
```

Base checkpoint:

\[
\approx3B
\]

### 3. Action Representation?

N1.7 emphasizes:

\[
\boxed{\text{relative end-effector action space}}
\]

shared across human and robot embodiments.

### 4. Training Objective?

Continuous action generation with Flow Matching.

### 5. Training Data?

Mixture of:

- real robot data
- simulation
- humanoid/bimanual data
- large-scale human video

N1.7 adds:

\[
20,000\text{ hours}
\]

of EgoScale human video pretraining.

### 6. Inference?

```text
language + image
→ VLM context

state + noisy actions
→ DiT conditioned on VLM

Flow Matching
→ continuous action chunk
→ execute selected horizon
```

---

# 26. GR00T N1.7 Engineering Details

Current public implementation documents:

- action horizon around 40
- expanded state/action interface
- ONNX export
- TensorRT deployment
- LeRobot integration
- multi-dataset fine-tuning support
- explicit execution-horizon terminology

These are highly relevant for training/deployment interviews.

---

# 27. Why Human Video Matters

Robot data is expensive; human video is abundant.

But human and robot actions are not naturally aligned.

GR00T N1.7 tries to reduce this gap using a shared relative EEF representation.

Conceptually:

```text
human manipulation motion
          │
          ▼
relative task-space motion
          │
          ▼
robot EEF control
```

Larger idea:

\[
\boxed{\text{use non-robot embodied data to improve robot priors}}
\]

---

# 28. GR00T vs π0 vs RDT

| | π0 | RDT | GR00T N1.7 |
|---|---|---|---|
| Semantic base | PaliGemma | T5 + SigLIP | Cosmos-Reason2 / Qwen3-VL |
| Action generator | Flow expert | diffusion Transformer | Flow-Matching DiT |
| Main target | general manipulation | multi-robot/bimanual | humanoid/cross-embodiment |
| Action representation | continuous | unified continuous | relative EEF-centered |
| Human video emphasis | low in original π0 | not central | strong |
| Deployment ecosystem | PI stack | research code | Isaac + TensorRT + LeRobot |

---

# 29. WALL-OSS — Tighter Coupling Between VLM and Action Learning

WALL-OSS asks:

> A web-pretrained VLM may understand images and language, but does it understand embodied task progress, affordances, and action?

## Six-Question Summary

### 1. Input?

- images
- instruction
- robot state
- optionally reasoning/subtask context

### 2. Backbone?

Qwen2.5-VL-3B-based model.

Architecture:

```text
shared self-attention
+
vision-language FFN
+
action FFN
```

Static routing selects specialized FFNs.

### 3. Action Representation?

Training uses both:

- FAST discrete action tokens
- continuous actions with Flow Matching

### 4. Training Objective?

Multi-stage combination of:

```text
embodied/general VQA
+
FAST next-token prediction
+
continuous Flow Matching
```

### 5. Training Data?

Mixture of:

- self-collected robot data
- open robot datasets
- general VQA
- embodied VQA
- localization/task-progress/affordance supervision

### 6. Inference?

Can produce:

```text
instruction
→ reasoning / subtask
→ continuous action
```

or use a more direct instruction-to-action path.

---

# 30. WALL-OSS Training Stages

Simplified view:

```text
Qwen2.5-VL
   ↓
Stage 1:
embodied/general VQA + FAST actions
   ↓
Stage 2:
freeze VLM, train continuous action branch
   ↓
Stage 3:
jointly train VLM + action branch
   ↓
task-specific adaptation
```

This staged recipe is relevant engineering-wise:

> Do not immediately unfreeze a giant VLM while a randomly initialized action head is unstable.

---

# 31. WALL-OSS vs π0.5

### π0.5

Main goal:

\[
\text{open-world long-horizon generalization}
\]

Mechanism:

```text
heterogeneous data
+
semantic subtask prediction
+
FAST pretraining
+
flow control
```

### WALL-OSS

Main goal:

\[
\text{make VLM representations more embodied}
\]

Mechanism:

```text
embodied VQA
+
shared multimodal attention
+
specialized FFNs
+
semantic reasoning
+
flow control
```

---

# 32. The Three Dominant Modern Action Paradigms

## Family A — Discrete Autoregressive Actions

Examples:

- RT-2
- OpenVLA
- π0-FAST

Pipeline:

```text
continuous actions
→ tokens
→ autoregressive Transformer
→ CE
```

Strengths:

- reuse VLM machinery
- easy text/action co-training

Weaknesses:

- quantization
- long sequences
- AR latency

---

## Family B — Diffusion Continuous Actions

Examples:

- Diffusion Policy
- Octo action head
- RDT

Pipeline:

```text
Gaussian noise
→ denoising network
→ continuous trajectory
```

Strengths:

- multimodal
- continuous
- good trajectory modeling

Weakness:

- iterative sampling cost

---

## Family C — Flow-Matching Continuous Actions

Examples:

- π0
- π0.5 runtime
- SmolVLA
- GR00T N1.7
- WALL-OSS continuous branch

Pipeline:

```text
noise
→ velocity field
→ ODE integration
→ continuous trajectory
```

Strengths:

- continuous precision
- natural chunk generation
- common in modern VLA systems

---

# 33. Second Major Axis: Where Does Semantics Live?

## Weak semantic backbone

ACT / Diffusion Policy:

```text
vision encoder
→ action policy
```

## Robot-specific language conditioning

RT-1:

```text
instruction embedding
→ visual features
→ robot Transformer
```

## Large pretrained VLM

RT-2 / OpenVLA:

```text
web-pretrained VLM
→ action tokens
```

## VLM + dedicated action expert

π0 / SmolVLA / GR00T:

```text
VLM
→ semantic context
→ continuous action model
```

## Explicit embodied-semantic training

π0.5 / WALL-OSS:

```text
VLM
→ subtask / reasoning
→ action policy
```

---

# 34. Third Major Axis: How Are Robot Embodiments Unified?

Methods include:

### Padding / masks

```text
largest state/action dimension
smaller robot
→ padding + mask
```

### Shared task-space representation

Example:

```text
relative end-effector delta
```

### Physically interpretable unified vector

RDT:

```text
fixed semantic slot for each physical quantity
```

### Embodiment-specific adapters

Common backbone with robot-specific I/O transforms.

### Interview point

Cross-embodiment learning requires alignment of:

- dimensionality
- units
- coordinate frames
- control semantics
- frequency
- gripper meaning
- missing joints/modalities

Not just tensor padding.

---

# 35. Full Architecture Comparison Table

| Model | Semantic Backbone | Action Representation | Action Generator | Main Loss |
|---|---|---|---|---|
| ACT | task visual encoder | continuous chunk | Transformer decoder | L1 + KL |
| Diffusion Policy | visual encoder | continuous chunk | diffusion denoiser | noise MSE |
| RT-1 | USE + FiLM EfficientNet | discretized vector | Transformer | CE |
| RT-2 | PaLI-X / PaLM-E | action tokens | VLM AR decoder | CE |
| Octo | T5 + robot Transformer | continuous chunk | diffusion head | diffusion |
| OpenVLA | Prismatic VLM | action tokens | VLM AR decoder | CE |
| π0 | PaliGemma | continuous chunk | flow action expert | Flow Matching |
| π0-FAST | VLM | FAST tokens | AR decoder | CE |
| π0.5 | VLM + action expert | FAST + continuous | AR + flow expert | CE + FM |
| SmolVLA | SmolVLM-2 | continuous chunk | flow expert | FM |
| RDT | T5 + SigLIP | continuous chunk | large DiT | diffusion |
| GR00T N1.7 | Cosmos-Reason2/Qwen3-VL | relative continuous | FM DiT | FM |
| WALL-OSS | Qwen2.5-VL | FAST + continuous | AR + flow branch | CE + FM + semantic |

---

# 36. Training-Engineer Comparison Table

| Model | VLM Train Strategy | Separate Action Module? | Cross-Embodiment? | Main Engineering Issue |
|---|---|---:|---:|---|
| ACT | N/A | policy itself | weak | precise chunking |
| Diffusion Policy | N/A | denoiser | weak | sampling latency |
| RT-2 | co-fine-tune | no continuous expert | limited | huge VLM + AR |
| Octo | robot backbone pretraining | diffusion head | strong | dataset heterogeneity |
| OpenVLA | full/LoRA adaptation | token output | strong | 7B training cost |
| π0 | VLM + expert | yes | strong | flow sampling + huge data |
| π0.5 | staged/mixed | yes | strong | heterogeneous objectives |
| SmolVLA | VLM frozen | yes | moderate | small-data efficiency |
| RDT | pretrained encoders + DiT | DiT is main model | strong | unified action space + DeepSpeed |
| GR00T | VLM + DiT pipeline | yes | very strong | data + deployment system |
| WALL-OSS | freeze then unfreeze | yes | strong | multi-objective curriculum |

---

# 37. Model Evolution: What Problem Did Each Solve?

### ACT

Problem:

> one-step BC is bad for precise high-frequency manipulation.

Solution:

> action chunks + temporal ensemble.

### Diffusion Policy

Problem:

> deterministic BC cannot model multimodal actions well.

Solution:

> diffusion over action trajectories.

### RT-1

Problem:

> robot policies do not scale well across many tasks.

Solution:

> language-conditioned Transformer + large real-robot data.

### RT-2

Problem:

> robot-only data lacks semantic world knowledge.

Solution:

> pretrained VLM + action tokens + web/robot co-training.

### Octo

Problem:

> every new robot requires a new policy.

Solution:

> cross-embodiment pretraining + modular token interfaces.

### OpenVLA

Problem:

> strong VLAs were mostly closed and huge.

Solution:

> open pretrained VLM + OXE + action-token training.

### π0

Problem:

> discrete action tokens are awkward for high-frequency dexterous control.

Solution:

> continuous Flow Matching action expert.

### FAST

Problem:

> autoregressive high-frequency chunks create too many redundant tokens.

Solution:

> DCT + quantization + BPE compression.

### π0.5

Problem:

> VLA struggles with long-horizon unseen homes/tasks.

Solution:

> heterogeneous data + semantic subtasks + FAST pretraining + flow runtime.

### SmolVLA

Problem:

> VLA training/deployment is too expensive.

Solution:

> small frozen VLM + compact flow expert + async runtime.

### RDT

Problem:

> diffusion robot policies do not scale cleanly across diverse embodiments.

Solution:

> large DiT + physically interpretable unified action space.

### GR00T

Problem:

> humanoid learning needs huge heterogeneous embodied data and production tooling.

Solution:

> VLM + FM DiT + cross-embodiment data + human video + Isaac ecosystem.

### WALL-OSS

Problem:

> web VLMs are not sufficiently embodied.

Solution:

> embodied semantic supervision + tightly coupled VLM/action architecture.

---

# 38. Why Are Modern VLAs Adding Action Experts?

If we directly force robot actions through a language decoder:

1. actions are continuous
2. control is high-frequency
3. precision matters
4. robot dynamics differ from text statistics
5. AR decoding adds latency

So modern systems often separate:

```text
semantic model
    ↓
VLM representation
    ↓
continuous action expert
```

---

# 39. Why Keep the VLM?

Why not only use Diffusion Policy?

VLM pretraining supplies priors for:

- object identity
- language grounding
- semantic attributes
- unseen concepts
- spatial relationships
- visual generalization

Robot data teaches:

\[
\text{how to act}
\]

while VLM pretraining already teaches much of:

\[
\text{what is in the world}
\]

---

# 40. Frozen or Trainable VLM?

No universal answer.

## Freeze

Advantages:

- low compute
- stable semantics
- less forgetting

Example:

- SmolVLA

## Fine-tune

Advantages:

- adapt visual features to manipulation
- improve control-sensitive representations

Example:

- OpenVLA

## Freeze → Unfreeze

Advantages:

- stabilize the new action branch first
- later align perception and control

Example:

- WALL-OSS-style curriculum

### Interview answer

> Choose based on data scale, compute, domain gap, and action-head maturity rather than assuming the VLM should always be frozen.

---

# 41. Why Not Compare Models Only by Parameter Count?

Robot performance depends strongly on:

- demonstration quality
- embodiment diversity
- action representation
- control frequency
- chunk length
- runtime latency
- sensor setup
- task difficulty
- pretraining data
- post-training data

Therefore:

\[
\text{larger model}
\not\Rightarrow
\text{better robot policy}
\]

---

# 42. What You Should Memorize

For each model:

- conceptual contribution
- semantic backbone family
- action representation
- loss family
- one important data/training fact
- inference style

Do **not** spend much time memorizing:

- every benchmark score
- every ablation percentage
- all optimizer hyperparameters
- every hidden dimension

unless directly related to your project or target company.

---

# 43. Priority for VLA Training Interviews

## Tier A — Must Be Very Strong

- Diffusion Policy
- OpenVLA
- π0
- π0.5
- Flow Matching
- action chunking
- RDT
- GR00T

You should handle follow-ups.

## Tier B — Must Explain Clearly

- ACT
- RT-1
- RT-2
- Octo
- FAST
- SmolVLA
- WALL-OSS

## Tier C — Know the Relationship

- π0.6 / RECAP
- Fast-WAM
- MolmoAct2
- world models
- RL post-training

These will be revisited later.

---

# 44. Connect Week 2 to Your RM65 Project

Your project can be explained as a modern VLA architecture choice rather than only “modified π0.5”.

High-level structure:

```text
images + language + robot state
             │
             ▼
       multimodal backbone
             │
             ▼
          Plan tokens
             │
       ┌─────┼─────────┐
       ▼     ▼         ▼
      Box   FAST   continuous action
                      │
                      ▼
                Flow Matching
```

---

# 45. Your Project vs π0

π0:

```text
observation
→ continuous action expert
```

Your model:

```text
observation
→ explicit Plan
→ continuous action expert
```

Motivation:

> introduce semantic/subtask structure before low-level execution.

---

# 46. Your Project vs π0.5

π0.5:

```text
overall instruction
→ semantic subtask
→ low-level action
```

Your model similarly emphasizes plan conditioning, but uses explicit frame/subtask-level supervision and auxiliary target representations.

Be prepared to explain:

- whether Plan is autoregressive
- whether Plan uses ground truth during training
- exposure bias at inference
- how Plan tokens are visible to action tokens

---

# 47. Your Project vs WALL-OSS

WALL-OSS:

```text
semantic reasoning / subtask
→ action
```

Your project:

```text
Plan
→ Box / FAST / continuous action
```

Potential interview question:

> Why not train direct image-language-to-action only?

Possible answer:

> Explicit intermediate supervision can encourage the model to learn task stage and object-level intent instead of asking the continuous action branch to discover all semantic structure implicitly.

---

# 48. Your Auxiliary FAST Branch

Why predict both:

```text
FAST
and
continuous actions?
```

Potential rationale:

- FAST supplies discrete trajectory-level supervision
- continuous branch preserves precision
- plan-conditioned multi-task supervision may improve shared representations
- auxiliary branch can be removed at runtime

But be ready for:

> Could the auxiliary loss hurt?

Yes.

Possible issue:

- gradient interference
- competing representations
- poor loss weighting

---

# 49. Whiteboard Drill

Draw this without notes:

```text
RT-2 / OpenVLA:

image + instruction
        ↓
       VLM
        ↓
action token 1
        ↓
action token 2
        ↓
...
        ↓
continuous action


π0:

image + instruction
        ↓
       VLM
        │
state + noisy actions
        ↓
  action expert
        ↓
velocity field
        ↓
Flow integration
        ↓
continuous action chunk
```

Target time:

\[
<3\text{ minutes}
\]

---

# 50. Architecture Identification Drill

Given:

```text
SigLIP + T5
→ large DiT
→ 64-step continuous action
```

Answer:

> RDT-like architecture.

Given:

```text
Qwen-style VLM
→ shared attention
→ separate action FFN
→ FAST pretraining
→ Flow Matching
```

Answer:

> WALL-OSS-like architecture.

Given:

```text
small frozen VLM
→ Flow Matching expert
→ async action queue
```

Answer:

> SmolVLA-like architecture.

---

# 51. Interview Questions — Basic

### Q1
Why is ACT relevant to modern VLA architectures?

### Q2
What did Diffusion Policy add beyond normal BC?

### Q3
What does TokenLearner do in RT-1?

### Q4
What is the conceptual difference between RT-1 and RT-2?

### Q5
Why does RT-2 co-train with web data?

### Q6
Why is Octo not simply an OpenVLA-like model?

### Q7
What is OpenVLA's action representation?

### Q8
Why does π0 use a separate action expert?

### Q9
Is FAST a VLA model?

### Q10
What changes from π0 to π0.5?

---

# 52. Interview Questions — Intermediate

### Q11
Compare OpenVLA and π0.

### Q12
Compare Diffusion Policy and RDT.

### Q13
Compare π0 and RDT.

### Q14
Compare π0 and SmolVLA.

### Q15
What is the purpose of RDT's unified action space?

### Q16
Why does GR00T emphasize relative EEF actions?

### Q17
Why might human video improve a robot VLA?

### Q18
Why would a model first freeze and later unfreeze its VLM?

### Q19
What is the difference between semantic planning and action generation?

### Q20
Why is high-level reasoning not automatically useful for robot success?

---

# 53. Interview Questions — Advanced

### Q21
You have 500 robot trajectories and one 24 GB GPU. Which architecture family would you start from and why?

Discuss:

- checkpoint availability
- full fine-tuning vs frozen VLM
- LoRA
- model size
- runtime hardware
- data diversity

Do not answer with only a model name.

### Q22
Your task requires 50 Hz control. Would you prefer autoregressive action tokens or continuous chunks?

Discuss:

- latency
- chunk size
- async inference
- closed-loop replanning
- precision

### Q23
Your robot has a different action space from the pretrained model. What do you change?

Possible areas:

- action adapter
- coordinate frame
- normalization
- dimensionality
- masks
- embodiment tag
- control frequency
- state representation

### Q24
A pretrained VLA recognizes the correct object but grasps badly. Is the failure semantic or control?

Likely:

```text
semantic grounding may be correct
but pose/action/calibration/control may fail
```

### Q25
A policy works on seen kitchens but fails in a new home. What could be wrong?

Investigate:

- visual generalization
- camera viewpoint
- calibration
- object geometry
- task decomposition
- recovery behavior
- training distribution

---

# 54. 10-Minute Mock Interview

## Part 1 — Evolution

> Walk me through ACT to π0.

Target:

```text
ACT
→ chunking

Diffusion Policy
→ multimodal continuous trajectories

RT-1
→ scalable language-conditioned robot Transformer

RT-2
→ web VLM knowledge + actions as tokens

OpenVLA
→ open VLM-based VLA

π0
→ continuous Flow Matching action expert
```

## Part 2 — Current Models

> What are RDT and GR00T doing differently?

Mention:

RDT:

- diffusion Transformer
- T5/SigLIP conditioning
- unified physical action space

GR00T:

- VLM + FM DiT
- humanoid/cross-embodiment
- human video
- relative EEF action representation
- deployment ecosystem

## Part 3 — Your Project

> Why did you add Plan tokens to π0.5?

Answer from:

- long-horizon structure
- task-stage supervision
- action conditioning
- error analysis
- auxiliary supervision

Then expect:

> What if Plan is wrong?

Prepare that answer too.

---

# 55. Self-Test Matrix

Score:

- `0`: cannot answer
- `1`: recognize after notes
- `2`: explain independently
- `3`: compare, critique, handle follow-ups

| Model / Topic | Score |
|---|---:|
| ACT | |
| Diffusion Policy | |
| RT-1 | |
| RT-2 | |
| Octo | |
| OpenVLA | |
| π0 | |
| FAST | |
| π0.5 | |
| SmolVLA | |
| RDT | |
| GR00T | |
| WALL-OSS | |
| discrete vs continuous actions | |
| diffusion vs flow | |
| cross-embodiment action space | |
| frozen vs trainable VLM | |
| semantic planner vs controller | |
| runtime implications | |
| RM65 project mapping | |

Target:

\[
\text{average}\ge2.3
\]

---

# 56. Final One-Page Review

```text
ACT
- continuous action chunks
- Transformer CVAE
- L1 + KL
- temporal ensemble

Diffusion Policy
- continuous chunks
- diffusion denoising
- multimodal actions

RT-1
- language-conditioned robot Transformer
- FiLM EfficientNet + TokenLearner
- discrete action bins

RT-2
- pretrained VLM
- actions as text-like tokens
- web + robot co-finetuning

Octo
- cross-embodiment robot foundation policy
- flexible tokenizers
- diffusion action head

OpenVLA
- open 7B VLA
- Prismatic VLM
- 7D discretized action tokens
- OXE pretraining

π0
- PaliGemma + ~300M action expert
- continuous 50-step chunks
- Flow Matching

FAST
- action tokenizer, not standalone VLA
- normalization + DCT + quantization + BPE

π0.5
- heterogeneous pretraining
- FAST pretraining + Flow Matching runtime
- semantic subtask hierarchy

SmolVLA
- ~450M
- frozen SmolVLM
- Flow Matching action expert
- affordable / async inference

RDT
- ~1.2B Diffusion Transformer
- T5 + SigLIP
- 64-step chunks
- unified physically interpretable action space

GR00T N1.7
- ~3B
- Cosmos-Reason2 / Qwen3-VL
- Flow-Matching DiT
- relative EEF actions
- human video + robot data
- Isaac / TensorRT / LeRobot

WALL-OSS
- Qwen2.5-VL
- shared attention + specialized FFNs
- embodied VQA
- FAST → Flow Matching curriculum
- reasoning/subtask → continuous control
```

---

# 57. Recommended Repository Note

Suggested path:

```text
docs/VLA_Interview/02_VLA_Architecture_Evolution.md
```

Do not replace your detailed individual paper notes.

Use this file as:

> the cross-paper interview index.

Your detailed model notes answer:

> “How does this paper work?”

This Week 2 note answers:

> “Why is this model different from the others?”

---

# 58. Primary References

Your existing notes:

- `docs/Model_Zoo/Robotics/Policies/ACT.md`
- `docs/Model_Zoo/Robotics/Policies/Diffusion_Policy.md`
- `docs/Model_Zoo/Robotics/Policies/RT_1.md`
- `docs/Model_Zoo/Robotics/Policies/RT_2.md`
- `docs/Model_Zoo/Robotics/Policies/Octo.md`
- `docs/Model_Zoo/Robotics/Policies/OpenVLA.md`
- `docs/Model_Zoo/Robotics/Policies/Pi_0.md`
- `docs/Model_Zoo/Robotics/Policies/Pi_0_FAST.md`
- `docs/Model_Zoo/Robotics/Policies/Pi_0_5.md`
- `docs/Model_Zoo/Robotics/Policies/SmolVLA.md`
- `docs/Model_Zoo/Robotics/Policies/WALL_OSS.md`

Missing from the current repo and worth adding separately later:

## RDT

- Paper: https://arxiv.org/abs/2410.07864
- Official code: https://github.com/thu-ml/RoboticsDiffusionTransformer
- Project: https://rdt-robotics.github.io/rdt-robotics/

## NVIDIA GR00T

- Official code: https://github.com/NVIDIA/Isaac-GR00T
- Current main branch: GR00T N1.7
- Platform: https://developer.nvidia.com/isaac/gr00t

---

# 59. Before Moving to Week 3

You are ready when you can do these without notes:

- [ ] explain ACT → π0 evolution
- [ ] distinguish RT-1 from RT-2
- [ ] distinguish Octo from OpenVLA
- [ ] compare OpenVLA vs π0
- [ ] explain why FAST is not a VLA
- [ ] explain π0 vs π0.5
- [ ] compare π0 / RDT / GR00T
- [ ] explain cross-embodiment action alignment
- [ ] explain frozen vs trainable VLM trade-offs
- [ ] draw at least three VLA architectures
- [ ] map your RM65 model into the architecture landscape
- [ ] answer all 25 interview questions orally

Week 3 will move away from paper architectures and into the part that matters most for a **VLA training engineer**:

\[
\boxed{
\text{raw robot trajectory}
\rightarrow
\text{dataset}
\rightarrow
\text{batch}
\rightarrow
\text{training step}
\rightarrow
\text{distributed training}
}
\]

That is where model knowledge becomes engineering ability.
