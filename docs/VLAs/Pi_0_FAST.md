# [FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)

## Convenient Links

* [Paper (arXiv)](https://arxiv.org/abs/2501.09747)
* [Project Page](https://pi.website/research/fast)
* [FAST+ Tokenizer](https://huggingface.co/physical-intelligence/fast)
* [pi0 note in this repo](./Pi_0.md)
* [OpenVLA note in this repo](./OpenVLA.md)
* [Diffusion Policy note in this repo](./Diffusion_Policy.md)
* [Real-Time Chunking note in this repo](./Real_Time_Chunking.md)

## 1. One-Sentence Summary

FAST is a **DCT + BPE action tokenizer** for high-frequency robot action chunks; when combined with the pi0 VLA backbone, it produces **pi0-FAST**, an autoregressive VLA that matches diffusion pi0 on dexterous robot tasks while reducing training compute by up to **5x**.

## 2. Why FAST Matters

Autoregressive VLAs such as RT-2 and OpenVLA can reuse standard next-token prediction:

```text
image + instruction -> VLM -> action tokens
```

But they need a discrete tokenization of continuous robot actions.

The common approach is simple per-dimension binning:

```text
continuous action dimension
-> one of 256 bins
-> one action token
```

This works tolerably for low-frequency control, but breaks down for high-frequency dexterous tasks:

* adjacent action timesteps are highly correlated
* action chunks can become hundreds of tokens long
* next-token prediction can learn trivial copying instead of meaningful control
* autoregressive decoding becomes slow
* training on datasets such as DROID becomes difficult

FAST attacks the tokenization problem directly:

> Compress the continuous action chunk before turning it into discrete tokens.

This makes autoregressive VLA training viable for dexterous, high-frequency robot data.

## 3. Problem Setup

The policy predicts a future action chunk:

$$
\pi(a_{1:H} \mid o)
$$

where:

$$
a_{1:H}
=
\left[
a_1,\,
a_2,\,
\dots,\,
a_H
\right]
$$

and each action is:

$$
a_t \in \mathbb{R}^{D}
$$

The action tokenizer maps a continuous action chunk to discrete tokens:

$$
T_a:
a_{1:H}
\rightarrow
\left[
T_1,\,
\dots,\,
T_n
\right],
\qquad
T_i \in \mathcal{V}
$$

Unlike naive binning, FAST allows:

$$
n \ll H \cdot D
$$

because it compresses the action trajectory before tokenization.

## 4. Why Naive Binning Fails

Naive VLA tokenization discretizes every timestep and every action dimension independently.

For a `D`-dimensional action chunk of length `H`, it produces:

$$
H \cdot D
$$

tokens:

$$
T_a(a_{1:H})
=
\left[
T_{1,1},\,
\dots,\,
T_{1,D},\,
\dots,\,
T_{H,1},\,
\dots,\,
T_{H,D}
\right]
$$

At high control frequency, consecutive actions change only slightly:

$$
a_t \approx a_{t+1}
$$

So the marginal information in the next token becomes small:

$$
I(T_i ; \text{future behavior} \mid T_{<i})
$$

is weak, because the next token is often predictable from the previous token.

The model can get low token loss by learning local smoothness:

```text
copy / slightly adjust the previous action token
```

instead of learning the global shape of the action chunk.

This is the paper's key diagnosis: high-frequency action data is redundant, so action tokens should be compressed before autoregressive learning.

## 5. FAST Tokenization Pipeline

FAST stands for **Frequency-space Action Sequence Tokenization**.

The pipeline is:

```text
raw action chunk
-> quantile normalize each action dimension
-> apply DCT along time
-> scale and round DCT coefficients
-> flatten low-frequency coefficients first
-> compress integer sequence with BPE
-> output action tokens
```

This is inspired by compression methods such as JPEG:

* low-frequency coefficients capture the broad trajectory shape
* high-frequency coefficients capture sharp changes
* smooth signals can be represented with few coefficients

Robot action chunks are often smooth, so DCT is a natural fit.

## 6. Quantile Normalization

FAST first normalizes each action dimension using training-set quantiles.

For action dimension `j`, let:

$$
q_{0.01}^{(j)}
\quad \text{and} \quad
q_{0.99}^{(j)}
$$

be the 1st and 99th percentiles.

The normalized value can be viewed as mapping:

$$
q_{0.01}^{(j)} \mapsto -1,
\qquad
q_{0.99}^{(j)} \mapsto 1
$$

The point is:

* handle different action scales across robots
* reduce sensitivity to outlier actions
* make cross-embodiment tokenization easier

This is similar in spirit to OpenVLA's use of quantile bounds, but FAST applies it before frequency-domain compression.

## 7. Discrete Cosine Transform

For each action dimension independently, FAST applies the DCT along the time axis:

$$
C^{(j)}
=
\operatorname{DCT}
\left(
a_{1:H}^{(j)}
\right)
$$

where:

* $a_{1:H}^{(j)}$ is the time series for action dimension `j`
* $C^{(j)}$ is the frequency-domain coefficient vector

Low-frequency coefficients describe the main shape:

```text
move smoothly upward
turn gradually
close gripper over this interval
```

High-frequency coefficients describe rapid changes:

```text
jerks
sharp contacts
quick corrections
```

For many robot actions, most information is concentrated in the low-frequency coefficients.

## 8. Quantizing DCT Coefficients

After DCT, FAST scales and rounds coefficients:

$$
\bar{C}^{(j)}_k
=
\operatorname{round}
\left(
\gamma C^{(j)}_k
\right)
$$

where:

$$
\gamma
$$

is the rounding scale.

The scale controls the compression-fidelity tradeoff:

| Larger `gamma` | Smaller `gamma` |
| :------------- | :-------------- |
| higher reconstruction fidelity | more zeros and more compression |
| more tokens | fewer tokens |
| more precise control | more lossy action reconstruction |

The paper reports that FAST is not very sensitive to this hyperparameter and uses:

```text
rounding scale = 10
BPE vocab size = 1024
```

for single-dataset tokenization experiments.

## 9. Flattening Order

The DCT coefficient matrix has shape:

$$
D \times H
$$

FAST flattens it into a 1D integer sequence before BPE.

The paper finds that flattening order matters. FAST uses **low-frequency-first** ordering:

```text
all low-frequency components across action dimensions
-> then higher-frequency components
```

This is better than listing all frequencies for dimension 1, then all frequencies for dimension 2, etc.

The reason is autoregressive:

> Predicting the low-frequency trajectory shape first gives the model a stable global plan before it fills in higher-frequency detail.

## 10. BPE Compression

After DCT quantization, many coefficients are zero or repeat in common patterns.

FAST applies byte-pair encoding (BPE) to the flattened integer sequence:

$$
\left[
T_1,\,
\dots,\,
T_k
\right]
\xrightarrow{\mathrm{BPE}}
\left[
\bar{T}_1,\,
\dots,\,
\bar{T}_{\bar{k}}
\right]
$$

BPE is lossless with respect to the quantized integer sequence. It compresses repeated patterns such as:

```text
0, 0, 0
low-frequency coefficient patterns across dimensions
common action-shape fragments
```

This gives FAST two levels of compression:

1. **lossy compression** from DCT coefficient rounding
2. **lossless compression** from BPE over the rounded sequence

## 11. FAST+ Universal Tokenizer

FAST has one learned component:

```text
BPE vocabulary
```

Training a BPE vocabulary is fast, but doing it per dataset adds friction.

So the paper trains **FAST+**, a universal robot action tokenizer, on approximately:

```text
1M one-second real robot action chunks
```

The tokenizer mixture covers:

* single-arm robots
* bimanual robots
* mobile manipulators
* joint-space actions
* end-effector actions
* camera-frame end-effector actions
* multiple control frequencies

Before tokenization, all actions are padded to:

```text
32 dimensions
```

so one tokenizer can cover different action spaces.

The released tokenizer can be loaded through Hugging Face:

```python
from transformers import AutoProcessor

tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast",
    trust_remote_code=True,
)

tokens = tokenizer(action_chunk)
```

## 12. FAST+ Training Mixture

The FAST+ mixture mostly comes from the pi0 dataset family, plus open-source robot data.

The appendix lists these groups:

| Group | Examples |
| :---- | :------- |
| bimanual | ARX, AgileX, Trossen biarm, ALOHA |
| single arm | Franka FR3, UR5, DROID, Bridge V2, OpenX |
| mobile | Fibocom, Mobile Trossen, ARX slate mobile |
| action spaces | joint, end-effector, camera-frame end-effector |
| frequencies | `5 Hz`, `15 Hz`, `20 Hz`, `50 Hz`, mixed |

Large mixture weights include:

| Dataset / group | Frequency | Mixture weight |
| :-------------- | :-------- | -------------: |
| DROID | `15 Hz` | `11.2%` |
| UR5 single joint | `20 Hz` | `10.3%` |
| ARX bimanual joint | `50 Hz` | `7.2%` |
| Bridge V2 | `5 Hz` | `5.0%` |
| ALOHA | `50 Hz` | `5.0%` |
| OpenX | mixed | `3.8%` |

This matters because FAST+ is meant to be a default tokenizer, not only a tokenizer for one robot arm.

## 13. pi0-FAST Policy Structure

pi0-FAST keeps the pi0-style VLA backbone but changes the action output interface.

The rough structure is:

```text
images
-> PaliGemma-style VLM encoder / language model

language instruction
-> text tokens

proprioceptive state
-> binned state tokens

FAST action tokens
-> autoregressive next-token targets
-> detokenize to continuous 1-second action chunk
```

The key difference from diffusion pi0:

| Model | Action decoder |
| :---- | :------------- |
| diffusion pi0 | flow / diffusion action expert |
| pi0-FAST | autoregressive language-model decoding of FAST action tokens |

So pi0-FAST does **not** need a separate flow-matching action expert. It uses the standard autoregressive next-token interface of the VLM.

## 14. Inputs and Outputs

The policy is conditioned on:

* `2` or `3` RGB images
* natural-language task instruction
* robot proprioceptive state

Appendix C states that the images are:

```text
224 x 224
```

and usually include:

* one third-person camera
* one wrist camera per robot arm

Each image is encoded separately through the pretrained vision encoder, and the visual tokens are concatenated.

The proprioceptive state is discretized into:

```text
256 bins
```

and then tokenized as part of the text input sequence.

The action target is a FAST token sequence representing a one-second action chunk.

## 15. Training Objective

pi0-FAST is trained with standard autoregressive next-token prediction.

Let the tokenized action chunk be:

$$
Y
=
\left[
y_1,\,
y_2,\,
\dots,\,
y_N
\right]
$$

The model predicts:

$$
p_\theta(Y \mid o, \ell, q)
=
\prod_{i=1}^{N}
p_\theta
\left(
y_i
\mid
o,\ell,q,y_{<i}
\right)
$$

where:

* $o$ are image observations
* $\ell$ is the language instruction
* $q$ is proprioceptive state
* $Y$ are FAST action tokens

The loss is:

$$
\mathcal{L}_{\mathrm{AR}}
=
-
\sum_{i=1}^{N}
\log
p_\theta
\left(
y_i^*
\mid
o,\ell,q,y_{<i}^*
\right)
$$

This is the central appeal of FAST:

> continuous robot action chunks can be trained with the same language-model objective used by VLMs.

## 16. General Policy Training Settings

The appendix gives the shared training recipe:

| Setting | Value |
| :------ | :---- |
| LR warmup | `1k` steps |
| LR after warmup | constant `5e-5` |
| Optimizer | AdamW |
| Adam betas | `beta1 = 0.9`, `beta2 = 0.95` |
| Weight decay | none |
| Gradient clipping | `1.0` |
| EMA weight | `0.999` |
| Inference decoding | greedy |
| Bimanual inference exception | temperature `0.7` |

The bimanual temperature is used for T-shirt folding, toast out of toaster, and laundry folding. The paper says it helps policies move out of the home position because some data contains stationary initial chunks.

## 17. DROID Training Setup

The DROID setup is especially important because prior autoregressive VLAs struggled to train useful zero-shot DROID policies.

Training data:

| Item | Setting |
| :--- | :------ |
| Episodes | `75k` successful episodes |
| Samples | `21M` samples |
| Training iterations | `240k` |
| Approximate epochs | `3` |
| Batch size | `256` |
| Hardware | `8 x H100` |
| Training time | about `4` days |

Inputs:

* one third-person view
* one wrist camera
* language instruction
* proprioceptive state

DROID provides two external camera views and three language annotations per episode. The paper randomly samples:

* the third-person view during training
* the language annotation during training

It does **not** use camera calibration.

Actions:

* joint velocity
* absolute gripper position
* `15`-step action chunks
* execute `8` or `15` steps open-loop at inference

Light curation:

* train only on successful episodes
* filter idle all-zero action timesteps

## 18. Evaluation Tasks

The paper evaluates FAST on `7` settings:

| Task | Domain | Frequency | Metric |
| :--- | :----- | :-------- | :----- |
| LIBERO | simulation | benchmark-dependent | average success across suites |
| DROID tabletop | real robot | `15 Hz` | task progress |
| Table bussing | UR5 single arm | `20 Hz` | percent objects sorted correctly |
| T-shirt folding | bimanual ARX | `50 Hz` | percent shirts folded |
| Grocery bagging | UR5 single arm | `20 Hz` | percent objects bagged |
| Toast out of toaster | bimanual Trossen | `50 Hz` | task progress out of 4 |
| Laundry folding | bimanual ARX | `50 Hz` | percent clothing folded and stacked |

The first four are used heavily for tokenizer comparisons. Grocery bagging, toaster, and laundry folding are used for the strongest generalist model comparisons.

## 19. Token Compression Results

FAST greatly reduces the number of action tokens per one-second chunk.

| Dataset | Action dim | Frequency | Naive tokens | FAST tokens | Compression |
| :------ | ---------: | --------: | -----------: | ----------: | ----------: |
| BridgeV2 | `7` | `5 Hz` | `35` | `20` | `1.75x` |
| DROID | `7` | `15 Hz` | `105` | `29` | `3.6x` |
| Bussing | `7` | `20 Hz` | `140` | `28` | `5.0x` |
| Shirt Fold | `14` | `50 Hz` | `700` | `53` | `13.2x` |

The high-frequency bimanual task is where FAST matters most:

```text
700 naive action tokens
-> 53 FAST action tokens
```

This is the core reason pi0-FAST can use autoregressive decoding for dexterous action chunks.

## 20. Tokenizer Comparison Results

The paper compares:

| Tokenizer | Description |
| :-------- | :---------- |
| Naive | per-dimension, per-timestep binning |
| FSQ | learned finite scalar quantization tokenizer |
| FAST | dataset-specific DCT + BPE tokenizer |
| FAST+ | universal DCT + BPE tokenizer |

Main results:

* naive tokenization struggles badly on high-frequency robot data
* naive policies make no meaningful progress on Table Bussing and T-shirt Folding
* compression-based tokenizers, FAST and FSQ, train much more effective policies
* FAST is as good as or better than FSQ, especially on dexterous real-robot tasks
* FAST+ closely matches dataset-specific FAST across tasks

The important practical conclusion:

> FAST+ can be used as a strong default tokenizer instead of retraining a tokenizer per robot dataset.

## 21. DROID Zero-Shot Results

The DROID result is one of the paper's strongest claims.

The policy is trained on DROID, then evaluated zero-shot in unseen environments:

* new scene background
* new camera angles
* new objects
* new table setup
* no co-training
* no fine-tuning
* prompted only with natural language

Quantitative evaluation:

| Item | Value |
| :--- | ----: |
| Tasks | `16` |
| Trials | `44` |
| Scoring | task-progress rubric |

Example tasks include:

* put the spoon in the dish rack
* put carrot in bowl
* wipe the table
* close the drawer
* clean the whiteboard
* put the marker in the cup
* move the watermelon from one bowl to another

The paper states that this is the first zero-shot evaluation of DROID policies in completely unseen environments without co-training or fine-tuning.

## 22. OpenVLA + FAST Ablation

The paper also tests whether FAST is specific to pi0.

It modifies OpenVLA to:

* accept multiple input images
* predict one-second action chunks
* use FAST+ action tokens

On the high-frequency T-shirt folding task, OpenVLA + FAST substantially improves over OpenVLA's original naive tokenization.

This supports the claim that FAST is a tokenizer improvement, not only a pi0-specific trick.

## 23. BPE Ablation

The paper removes BPE while keeping DCT.

Result:

* DCT without BPE still beats naive tokenization
* but it performs worse than full FAST
* it also slows inference because the model must decode many repeated zero tokens

Interpretation:

```text
DCT concentrates useful signal in low-frequency coefficients.
BPE removes repeated sparse patterns and shortens the sequence.
Both matter.
```

## 24. pi0-FAST vs. Diffusion pi0

The paper compares pi0-FAST to diffusion pi0 on single-task training.

Findings:

* on smaller datasets such as LIBERO and T-shirt folding, both are comparable
* on larger datasets such as Table Bussing, pi0-FAST converges faster
* on DROID, pi0-FAST follows language instructions more closely
* pi0-FAST reaches high performance on Table Bussing with about `3x` fewer training steps

However, pi0-FAST is slower at inference:

| Model | Inference behavior |
| :---- | :----------------- |
| diffusion pi0 | about `<100 ms` per one-second chunk on RTX 4090 |
| pi0-FAST | about `750 ms` per chunk |

The reason is architectural:

* diffusion pi0 runs a small `300M` action expert for about `10` diffusion steps
* pi0-FAST autoregressively decodes about `30-60` action tokens through the full `2B` language backbone

So the tradeoff is:

```text
pi0-FAST:
  faster training
  better use of autoregressive VLM objective
  slower inference
```

## 25. Scaling to Generalist Training

The final experiment trains pi0-FAST on the large cross-embodied pi0 data mixture.

The mixture contains:

| Data source | Amount |
| :---------- | :----- |
| Physical Intelligence robot data | `903M` timesteps |
| open-source data share | `9.1%` |
| open-source datasets | Bridge V2, DROID, Open X-Embodiment |
| total robot data scale | about `10k` hours |

The evaluation compares pi0-FAST to diffusion pi0 on the tasks from the pi0 paper.

Tasks include:

* T-shirt folding
* Table bussing
* Grocery bagging
* Toast out of toaster
* Laundry folding

Main result:

> pi0-FAST matches diffusion pi0 overall, including on the difficult laundry-folding task, while using about `5x` fewer GPU hours for training.

The paper also reports that when comparing against a compute-matched diffusion pi0 checkpoint, pi0-FAST clearly outperforms it because pi0-FAST converges faster.

## 26. What pi0-FAST Changes Relative to pi0

| Aspect | diffusion pi0 | pi0-FAST |
| :----- | :------------ | :------- |
| VLM backbone | PaliGemma-style VLM | PaliGemma-style VLM |
| Action representation | continuous action chunks | FAST action tokens |
| Action decoder | flow / diffusion action expert | autoregressive LM head |
| Training loss | flow matching | next-token cross-entropy |
| Training speed | slower | up to `5x` fewer GPU hours |
| Inference speed | faster | slower, about `750 ms` per chunk |
| Strength | smooth continuous action generation | efficient token-based VLA training |

This is the main architecture tradeoff.

FAST makes autoregressive VLAs viable for dexterous action chunks, but it does not automatically make autoregressive inference fast.

## 27. Limitations

The paper highlights several open problems:

* FAST+ is tested mostly on static robot manipulation datasets
* generalization to dexterous hands, humanoids, and mobile robots needs more study
* the best universal action tokenizer is not settled
* pi0-FAST inference is slower than diffusion pi0
* the best VLA architecture, autoregressive vs. diffusion, remains unresolved
* inference acceleration techniques such as speculative decoding, quantization, and custom kernels are left for future work

The key limitation for deployment is clear:

> pi0-FAST trains much faster, but diffusion pi0 currently runs faster at inference time.

## 28. Practical Takeaways

The main lessons are:

1. **Action tokenization is not a detail**
   * it can determine whether autoregressive VLA training works at all

2. **High-frequency robot data should be compressed**
   * per-timestep binning produces too many redundant tokens

3. **DCT is a strong prior for robot action chunks**
   * many robot motions are smooth enough to compress in frequency space

4. **BPE helps after DCT**
   * it removes repeated sparse coefficient patterns

5. **FAST+ is useful as a default**
   * one tokenizer can cover many robots and action spaces

6. **pi0-FAST changes the training/inference tradeoff**
   * faster training, slower inference

## 29. Mental Model

The shortest way to remember pi0-FAST is:

```text
Do not tokenize every action dimension at every timestep.
Compress the whole one-second action chunk with DCT.
Quantize the important frequency coefficients.
Use BPE to shorten repeated coefficient patterns.
Train the VLA with ordinary next-token prediction.
Detokenize the predicted tokens back into continuous actions.
```

FAST is the bridge that lets autoregressive VLMs treat dexterous continuous robot control more like language modeling, without the naive-token redundancy that breaks high-frequency action prediction.
