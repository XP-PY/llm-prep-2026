# [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)

## Convenient Links

* [ ] [Paper (arXiv)](https://arxiv.org/abs/2506.01844)
* [ ] [Hugging Face Blog](https://huggingface.co/blog/smolvla)
* [ ] [Model Card](https://huggingface.co/lerobot/smolvla_base)
* [ ] [LeRobot SmolVLA Docs](https://huggingface.co/docs/lerobot/smolvla)
* [ ] [LeRobot Code](https://github.com/huggingface/lerobot)
* [ ] [OpenVLA note in this repo](./OpenVLA.md)
* [ ] [Octo note in this repo](./Octo.md)
* [ ] [ACT note in this repo](./ACT.md)
* [ ] [Diffusion Policy note in this repo](./Diffusion_Policy.md)
* [ ] [LIBERO note in this repo](../Datasets/LIBERO.md)

## 1. One-Sentence Summary

SmolVLA is a compact **450M-parameter VLA** from Hugging Face / LeRobot that combines a frozen **SmolVLM-2** backbone with a **flow-matching action expert**, trains on about **22.9k community-contributed robot episodes**, predicts continuous action chunks, and shows that a small open model can compete with much larger VLAs on LIBERO, Meta-World, SO100, and SO101 tasks.

## 2. Why SmolVLA Matters

Many early VLA systems prove that vision-language pretraining can help robot control, but they are still expensive:

* large model size
* large robot datasets
* expensive training runs
* GPU-heavy inference
* limited accessibility for small labs and hobbyist robotics

SmolVLA attacks a different bottleneck:

> Can a useful VLA be trained and deployed with affordable hardware and community robot data?

The paper matters because it puts efficiency and reproducibility at the center:

* **small model**: main model is about `0.45B` parameters
* **small robot dataset**: about `22.9k` episodes, far below OpenVLA-scale data
* **community data**: 481 Hugging Face / LeRobot datasets, mostly from SO100-style affordable arms
* **continuous control**: uses flow matching over action chunks instead of discrete action tokens
* **fast deployment**: introduces asynchronous inference to reduce robot idle time

So SmolVLA is not simply a smaller OpenVLA. It is an accessibility-oriented recipe for building usable robot policies from open hardware, open data, and open tooling.

## 3. High-Level Pipeline

The overall recipe is:

```text
community LeRobot datasets
-> clean task annotations and camera names
-> freeze compact SmolVLM-2 backbone
-> train flow-matching action expert
-> output continuous action chunks
-> optionally run async client-server inference
```

At inference time, the policy receives:

* one or more RGB camera views
* a language instruction
* robot sensorimotor state

and outputs a future action chunk:

$$
A_t =
\left[
a_t,\,
a_{t+1},\,
\dots,\,
a_{t+n}
\right]
$$

The deployed controller then executes actions from this chunk, either synchronously or through the paper's asynchronous queue-based stack.

## 4. Architecture Overview

SmolVLA has two main components:

```text
SmolVLM-2 perception backbone
-> VLM features
-> flow-matching action expert
-> continuous action chunk
```

The split is important:

* the VLM handles perception and language grounding
* the action expert handles low-level continuous control

This is closer in spirit to [pi0](https://arxiv.org/abs/2410.24164) and diffusion-style action decoders than to OpenVLA's discrete action-token prediction.

## 5. VLM Backbone: SmolVLM-2

SmolVLA uses **SmolVLM-2** as the pretrained vision-language backbone.

The backbone includes:

* **SigLIP** vision encoder
* **SmolLM2** language decoder
* support for multi-image inputs

The inputs are converted into tokens:

```text
RGB images -> visual tokens
language instruction -> text tokens
robot state -> one projected state token
```

These tokens are concatenated and passed through the VLM decoder. The resulting features condition the action expert.

One practical nuance: the paper's implementation keeps the VLM frozen and trains the action expert. This makes the training recipe much cheaper than full VLA fine-tuning.

## 6. Visual Token Reduction

High-resolution visual inputs are expensive. SmolVLM-2 can use image tiling, but SmolVLA avoids tiling for speed.

The final design uses:

* global image only
* pixel shuffle operation
* **64 visual tokens per frame**
* images resized to `512 x 512`

This is a central efficiency choice. The model keeps enough visual context for manipulation while avoiding the heavy token budget of tiled high-resolution VLM inference.

## 7. Layer Skipping

SmolVLA does not always use the final VLM layer.

Instead, it discards the last `L - N` layers and feeds earlier VLM features to the action expert. In the main setup, the paper uses the first **16 layers** of the VLM language decoder.

Conceptually:

```text
full VLM has L layers
use features up to layer N
discard layers N+1 ... L
```

The reason is empirical and practical:

* later VLM layers are not always best for downstream control
* early or middle features can preserve useful spatial information
* skipping layers cuts latency and compute

In the ablation, using the first half of a larger VLM gives a good efficiency-performance tradeoff, and performs better than simply replacing the VLM with a much smaller one.

## 8. Flow-Matching Action Expert

The action expert predicts continuous action chunks using flow matching.

Let the clean action chunk be:

$$
A_t =
\left[
a_t,\,
\dots,\,
a_{t+n}
\right]
$$

The noisy interpolation is:

$$
A_t^\tau
=
\tau A_t
+
(1 - \tau)\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I)
$$

The action expert learns a vector field:

$$
v_\theta(A_t^\tau, h_t, \tau)
$$

where $h_t$ are the VLM features. The target vector field is:

$$
u(A_t^\tau \mid A_t)
=
\epsilon - A_t
$$

The training objective is:

$$
\mathcal{L}_{\text{FM}}
=
\mathbb{E}
\left[
\left\|
v_\theta(A_t^\tau, h_t, \tau)
-
u(A_t^\tau \mid A_t)
\right\|_2^2
\right]
$$

At inference time, the paper uses **10 flow-matching steps**.

This objective is important because continuous robot actions are not naturally language tokens. SmolVLA avoids discretizing actions into bins and instead learns a continuous generative control model.

## 9. Interleaved Cross-Attention and Self-Attention

The action expert uses Transformer blocks, but the paper does not use only one attention style.

It interleaves:

* **cross-attention (CA)** from action tokens to VLM features
* **causal self-attention (SA)** among action tokens

The intuition is:

```text
cross-attention -> ground actions in visual/language context
self-attention  -> make the action chunk temporally coherent
```

The self-attention is causal, so each action token can attend only to previous action tokens in the chunk. This avoids leaking future action information during training.

The paper reports that interleaving CA and SA improves success and helps produce smoother real-robot actions.

## 10. Community Pretraining Data

SmolVLA's data story is one of the paper's main contributions.

The pretraining set contains:

| Source             | Count |
| :----------------- | ----: |
| Community datasets |   481 |
| Episodes           | 22.9k |
| Frames             | 10.6M |

These datasets come from Hugging Face / LeRobot community contributions. Unlike carefully standardized academic robot datasets, they are messy:

* inconsistent task descriptions
* different camera names
* varied household or lab settings
* noisy demonstrations
* heterogeneous objects and tasks

The paper argues that this messiness is useful because it reflects the kind of data affordable robot users can actually collect.

## 11. Data Cleaning and Standardization

### 11.1 Task Annotation with a VLM

Many community datasets had weak task labels:

* missing instructions
* placeholders such as `task desc`
* vague labels such as `Hold` or `Up`

The authors use **Qwen2.5-VL-3B-Instruct** to generate concise action-oriented task descriptions from representative frames and original metadata.

The goal is not to create rich language supervision. It is to make each dataset's task instruction clear enough for language-conditioned policy training.

### 11.2 Camera View Normalization

Community datasets also use inconsistent camera names.

For example, a field name such as `images.laptop` may refer to a top, side, or wrist view depending on the contributor.

SmolVLA manually maps cameras into a consistent order:

```text
OBS_IMAGE_1 -> top view
OBS_IMAGE_2 -> wrist view
OBS_IMAGE_3 -> side view
```

Extra views are dropped during training.

This is a small but important engineering lesson: with small robot data, inconsistent camera ordering can hurt training enough that manual normalization is worth doing.

## 12. Training Details

The main implementation details are:

| Component               | Setting                                        |
| :---------------------- | :--------------------------------------------- |
| Main model size         | about`450M` parameters                       |
| Action expert size      | about`100M` parameters                       |
| VLM backbone            | SmolVLM-2                                      |
| VLM training            | frozen                                         |
| Trainable module        | action expert                                  |
| Pretraining steps       | `200k`                                       |
| Global batch size       | `256`                                        |
| Warmup                  | `100` steps                                  |
| LR schedule             | cosine,`1e-4` to `2.5e-6`                  |
| Optimizer               | AdamW                                          |
| Adam betas              | `beta1 = 0.9`, `beta2 = 0.95`              |
| Image size              | `512 x 512`                                  |
| Action chunk size       | `n = 50`                                     |
| Flow steps at inference | `10`                                         |
| Precision               | bfloat16                                       |
| Efficiency tools        | `torch.compile`, Hugging Face `accelerate` |

The paper reports that pretraining used **4 GPUs** for a large batch size, but the model is small enough to train on a single GPU. The total project consumed about **30k GPU hours**.

## 13. Asynchronous Inference

Modern action-chunking policies face a deployment tradeoff.

One option is synchronous inference:

```text
observe
-> predict a full action chunk
-> execute the whole chunk
-> observe again
```

This is compute-efficient, but the robot acts open-loop for the whole chunk and can become idle while waiting for the next prediction.

Another option is to predict at every control step, but that is expensive.

SmolVLA introduces a middle ground:

```text
RobotClient executes actions from a queue
PolicyServer predicts a new chunk in the background
new chunks are merged into the queue when ready
```

The key control parameter is a threshold `g`:

* when the remaining queue fraction drops below `g`, request a new chunk
* `g = 0` behaves like fully synchronous inference
* `g = 1` behaves like near-continuous prediction
* intermediate `g` balances reactivity and compute

The stack also filters near-duplicate observations using joint-space similarity, so the policy server does not waste calls on almost identical states.

This design is model-agnostic: it can be used with any policy that outputs action chunks.

## 14. Evaluation Setup

SmolVLA is evaluated in both simulation and real-world settings.

### 14.1 Simulation

The simulation benchmarks are:

| Benchmark  | Setup                                                      |
| :--------- | :--------------------------------------------------------- |
| LIBERO     | 40 tasks: Spatial, Object, Goal, Long, 10 tasks each       |
| Meta-World | 50 tasks with easy, medium, hard, and very hard categories |

For LIBERO:

* dataset: `physical-intelligence/libero`
* `1,693` episodes
* `10` trials per task
* success is binary

For Meta-World:

* dataset: `lerobot/metaworld_mt50`
* `2,500` episodes
* `50` demonstrations per task
* `10` trials per task
* success is binary

### 14.2 Real World

The real-world robot arms are:

* **SO100**: low-cost, open-source, 3D-printable arm
* **SO101**: updated SO100-style arm with smoother movements and faster assembly

The evaluated datasets are:

| Dataset                          | Robot | Task                                            |
| :------------------------------- | :---- | :---------------------------------------------- |
| `lerobot/svla_so100_pickplace` | SO100 | pick cube and place it in a box                 |
| `lerobot/svla_so100_stacking`  | SO100 | stack red cube on blue cube                     |
| `lerobot/svla_so100_sorting`   | SO100 | sort colored cubes into boxes                   |
| `lerobot/svla_so101_pickplace` | SO101 | place a small Lego brick into a transparent box |

Each real-world dataset has:

* `50` demonstrations
* `5` starting positions
* `10` trajectories per position

The SO101 evaluation is especially useful because SmolVLA is **not pretrained on SO101 data**, so it probes cross-embodiment adaptation.

## 15. Main Results: Simulation

### 15.1 LIBERO

| Policy           | Robotics VLA Pretraining | Spatial | Object | Goal | Long |   Avg |
| :--------------- | :----------------------: | ------: | -----: | ---: | ---: | ----: |
| Diffusion Policy |            No            |    78.3 |   92.5 | 68.3 | 50.5 |  72.4 |
| Octo 0.09B       |           Yes           |    78.9 |   85.7 | 84.6 | 51.1 |  75.1 |
| OpenVLA 7B       |           Yes           |    84.7 |   88.4 | 79.2 | 53.7 |  76.5 |
| pi0 Paligemma-3B |            No            |      87 |     63 |   89 |   48 |  71.8 |
| pi0 3.3B         |           Yes           |      90 |     86 |   95 |   73 |  86.0 |
| SmolVLA 0.24B    |            No            |      87 |     93 |   88 |   63 | 82.75 |
| SmolVLA 0.45B    |            No            |      90 |     96 |   92 |   71 |  87.3 |
| SmolVLA 2.25B    |            No            |      93 |     94 |   91 |   77 | 88.75 |

The key result is that **SmolVLA-0.45B reaches 87.3 average success**, slightly above the robotics-pretrained pi0 3.3B result and clearly above OpenVLA in this setup.

This does not mean SmolVLA is universally stronger than OpenVLA. It means that for this multitask simulation protocol, a compact continuous-action VLA can be extremely competitive.

### 15.2 Meta-World

| Policy           | Robotics VLA Pretraining |  Easy | Medium | Hard | Very Hard |   Avg |
| :--------------- | :----------------------: | ----: | -----: | ---: | --------: | ----: |
| Diffusion Policy |            No            |  23.1 |   10.7 |  1.9 |       6.1 |  10.5 |
| TinyVLA          |            No            |  77.6 |   21.5 | 11.4 |      15.8 |  31.6 |
| pi0 Paligemma-3B |            No            |  80.4 |   40.9 | 36.7 |      44.0 |  50.5 |
| pi0 3.5B         |           Yes           |  71.8 |   48.2 | 41.7 |      30.0 |  47.9 |
| SmolVLA 0.24B    |            No            | 86.43 |  46.36 |   35 |        60 | 56.95 |
| SmolVLA 0.45B    |            No            |  82.5 |   41.8 | 45.0 |      60.0 |  57.3 |
| SmolVLA 2.25B    |            No            | 87.14 |  51.82 |   70 |        64 | 68.24 |

On Meta-World, the larger SmolVLA variant is strongest, but the 0.45B model still beats the listed pi0 and TinyVLA baselines on average.

## 16. Main Results: Real Robots

### 16.1 SO100

| Policy        | Training    | Pick-Place | Stacking | Sorting |  Avg |
| :------------ | :---------- | ---------: | -------: | ------: | ---: |
| ACT           | single-task |         70 |       50 |      25 | 48.3 |
| pi0 3.5B      | multi-task  |        100 |       40 |      45 | 61.7 |
| SmolVLA 0.45B | multi-task  |         75 |       90 |      70 | 78.3 |

SmolVLA is not best on every task. pi0 is stronger on pick-place. But SmolVLA is much stronger on stacking and sorting, so its average is highest.

The sorting result is especially important because sorting is longer-horizon and requires color-conditioned behavior.

### 16.2 SO101

| Policy        | In Distribution | Out of Distribution |
| :------------ | --------------: | ------------------: |
| ACT           |              70 |                  40 |
| SmolVLA 0.45B |              90 |                  50 |

The SO101 task uses a different robot embodiment from the community SO100 pretraining set. SmolVLA still improves over ACT in both in-distribution and OOD object placements.

This supports the paper's claim that community pretraining plus fine-tuning can transfer beyond the exact robot used during pretraining.

## 17. Effect of Pretraining and Multitask Fine-Tuning

The real-world SO100 ablation is one of the clearest results in the paper:

| Policy        | Community Pretraining | Training    | Pick-Place | Stacking | Sorting |  Avg |
| :------------ | :-------------------: | :---------- | ---------: | -------: | ------: | ---: |
| SmolVLA 0.45B |          No          | single-task |         55 |       45 |      20 |   40 |
| SmolVLA 0.45B |          No          | multi-task  |         80 |       40 |      35 | 51.7 |
| SmolVLA 0.45B |          Yes          | multi-task  |         75 |       90 |      70 | 78.3 |

Two takeaways:

1. **Multitask fine-tuning helps**
   * average success rises from `40` to `51.7` without community pretraining
2. **Community pretraining helps much more**
   * average success rises from `51.7` to `78.3`

This is the central empirical argument for the paper's community-data strategy.

## 18. Asynchronous Inference Results

The paper compares synchronous and asynchronous inference on real SO100 tasks.

| Inference | Pick-Place | Stacking | Sorting |  Avg |
| :-------- | ---------: | -------: | ------: | ---: |
| Sync      |         75 |       90 |      70 | 78.3 |
| Async     |         80 |       90 |      50 | 73.3 |

Success remains broadly comparable, with a drop on sorting.

The speed gains are more striking:

| Inference | Total Time | Avg Time |  Std |
| :-------- | ---------: | -------: | ---: |
| Sync      |     137.5s |   13.75s | 2.42 |
| Async     |      97.0s |    9.70s | 2.95 |

In a fixed-time pick-place test:

| Inference | Total Cubes | Avg |  Std |
| :-------- | ----------: | --: | ---: |
| Sync      |           9 | 1.8 | 0.45 |
| Async     |          19 | 3.8 |  1.3 |

So async inference gives:

* about **30% faster** average completion time
* more than **2x** successful pick-place cycles under a fixed time budget

The practical lesson is that action-generation latency is not only a model-speed issue. Runtime scheduling and action queues can materially change real robot throughput.

## 19. Ablation Highlights

All ablations are run on LIBERO unless otherwise stated.

### 19.1 Cross-Attention and Self-Attention

| Attention | Spatial | Object | Goal | Long |  Avg |
| :-------- | ------: | -----: | ---: | ---: | ---: |
| CA        |      87 |     92 |   83 |   54 | 79.0 |
| SA        |      80 |     94 |   84 |   40 | 74.5 |
| CA + SA   |      86 |     99 |   90 |   67 | 85.5 |

Interleaving CA and SA is clearly best. CA grounds the chunk in observation features, while SA helps action tokens coordinate over time.

### 19.2 Causal Attention Beats Bidirectional Attention

| Action Attention Mask | Spatial | Object | Goal | Long |  Avg |
| :-------------------- | ------: | -----: | ---: | ---: | ---: |
| Bidirectional         |      79 |     86 |   82 |   23 | 67.5 |
| Causal                |      80 |     94 |   84 |   40 | 74.5 |

This is a useful detail. Letting future action tokens interact bidirectionally can hurt, likely because it creates training-time dependencies that do not match the generation process cleanly.

### 19.3 Flow Matching Beats Regression

| Objective     | Spatial | Object | Goal | Long |   Avg |
| :------------ | ------: | -----: | ---: | ---: | ----: |
| Flow matching |      89 |     94 |   85 |   53 | 80.25 |
| Regression    |      92 |     85 |   86 |   38 | 75.25 |

Regression is not terrible, but flow matching is stronger, especially on the long-horizon split.

This lines up with [Diffusion Policy](./Diffusion_Policy.md): continuous robot actions often benefit from generative objectives that can represent multimodal action distributions.

### 19.4 Chunk Size Matters

| Chunk Size | Spatial | Object | Goal | Long |  Avg |
| ---------: | ------: | -----: | ---: | ---: | ---: |
|          1 |      45 |     77 |   54 |   24 | 50.0 |
|         10 |      90 |     94 |   94 |   58 | 84.0 |
|         30 |      85 |     94 |   87 |   48 | 78.5 |
|         50 |      89 |     94 |   85 |   53 | 80.3 |
|        100 |      83 |     88 |   85 |   42 | 74.5 |

Very small chunks behave too much like one-step prediction. Very large chunks reduce reactivity.

The best range is around `10-50` actions.

### 19.5 Updating Observations More Often Helps

| Executed Actions Before Update | Spatial | Object | Goal | Long |  Avg |
| -----------------------------: | ------: | -----: | ---: | ---: | ---: |
|                              1 |      89 |     94 |   85 |   53 | 80.3 |
|                             10 |      89 |     94 |   91 |   57 | 82.8 |
|                             30 |      76 |     91 |   74 |   42 | 70.8 |
|                             50 |      54 |     70 |   58 |   25 | 51.8 |

This table explains why asynchronous inference matters. The model can predict long chunks, but executing too much of a chunk before refreshing observations hurts closed-loop robustness.

## 20. Why SmolVLA Works

SmolVLA's gains come from several aligned choices:

1. **Small but useful VLM backbone**
   * SmolVLM-2 provides multimodal grounding without a 7B-scale model.
2. **Aggressive token and layer efficiency**
   * 64 visual tokens per frame and layer skipping reduce latency.
3. **Continuous action generation**
   * flow matching avoids action-token discretization.
4. **Action expert separation**
   * the VLM can stay frozen while the action expert learns robot control.
5. **Community robot data**
   * 481 datasets give broad real-world variation despite small total scale.
6. **Runtime systems design**
   * async inference improves real robot responsiveness without requiring per-timestep full inference.

The broader lesson is that VLA performance is not only about model scale. Data accessibility, action representation, and deployment scheduling are equally important.

## 21. Relation to Other VLA and Robot Policy Papers

Compared with [OpenVLA](./OpenVLA.md):

| Aspect                | OpenVLA                                     | SmolVLA                                |
| :-------------------- | :------------------------------------------ | :------------------------------------- |
| Main size             | 7B                                          | 0.45B main model                       |
| Backbone              | Prismatic VLM                               | SmolVLM-2                              |
| Action representation | discrete action tokens                      | continuous flow-matching action chunks |
| Robot data            | large Open X-Embodiment subset              | smaller community LeRobot datasets     |
| Main emphasis         | open generalist VLA training and adaptation | affordable training and deployment     |

Compared with [Octo](./Octo.md):

* Octo is robot-policy-first and uses flexible tokenizers plus diffusion action heads.
* SmolVLA is VLM-backed and emphasizes small model size, community data, and async inference.

Compared with [ACT](./ACT.md):

* ACT predicts action chunks with a CVAE and is strong for task-specific imitation.
* SmolVLA adds language conditioning, VLM perception, community pretraining, and multitask transfer.

Compared with [Diffusion Policy](./Diffusion_Policy.md):

* Diffusion Policy is a general action-diffusion recipe for visuomotor control.
* SmolVLA uses a flow-matching action expert inside a VLA stack.

Compared with [LIBERO](../Datasets/LIBERO.md):

* LIBERO is a benchmark for controlled lifelong manipulation transfer.
* SmolVLA uses LIBERO as one of its main simulation evaluations and ablation testbeds.

## 22. Limitations

The paper is clear about remaining limitations:

1. **Pretraining data is mostly one embodiment**
   * The community pretraining data mainly comes from SO100-style robots.
2. **Dataset size is still small**
   * About `23k` trajectories is far smaller than OpenVLA-scale robot data.
3. **The VLM backbone may not be robot-optimal**
   * SmolVLM-2 is not specifically pretrained for robot interaction.
4. **Tasks are still relatively short-horizon**
   * Scaling to complex long-horizon manipulation may require planning or hierarchy.
5. **The method is mostly imitation learning**
   * Reinforcement learning or online correction could help for dexterous or failure-prone tasks.
6. **Async inference has tradeoffs**
   * It improves throughput, but success can drop on harder tasks such as sorting.

## 23. Key Takeaways

1. **SmolVLA shows that useful open VLAs do not need to be 7B+ models**
2. **The main 0.45B model combines SmolVLM-2 with a flow-matching action expert**
3. **Community robot data is small but valuable: 481 datasets, 22.9k episodes, 10.6M frames**
4. **Continuous action chunks are central to the model's control interface**
5. **Interleaved cross-attention and causal self-attention improve the action expert**
6. **Flow matching beats direct regression on LIBERO**
7. **Community pretraining and multitask fine-tuning strongly improve real-world SO100 results**
8. **Asynchronous inference is a practical systems contribution, giving faster robot throughput without changing the policy model**
