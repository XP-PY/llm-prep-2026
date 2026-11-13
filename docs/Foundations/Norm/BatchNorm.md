# [BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) and Batch Normalization

## 1. Why this note exists

`BatchNorm` is one of the most influential normalization methods in deep learning, especially in CNNs.

But it is easy to confuse with:

* `LayerNorm`
* `RMSNorm`

The key question is:

> what exactly is the "batch" in BatchNorm, and over which dimensions are the statistics computed?

The clean mental model is:

> BatchNorm normalizes one channel using statistics computed from the current mini-batch.

That already tells you why it works so well in CNNs and why it is less natural for autoregressive Transformers.

---

## 2. Core Formula

Suppose we have one feature channel with batch values:

$$
x_1, x_2, \dots, x_m
$$

BatchNorm first computes the batch mean:

$$
\mu_B = \frac{1}{m}\sum_{k=1}^{m}x_k
$$

and batch variance:

$$
\sigma_B^2 = \frac{1}{m}\sum_{k=1}^{m}(x_k-\mu_B)^2
$$

Then it standardizes:

$$
\hat{x}_k = \frac{x_k-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

and applies learnable affine parameters:

$$
y_k = \gamma \hat{x}_k + \beta
$$

So the full formula is:

$$
\operatorname{BatchNorm}(x_k)
=
\gamma \frac{x_k-\mu_B}{\sqrt{\sigma_B^2+\epsilon}} + \beta
$$

where:

* $\mu_B$ and $\sigma_B^2$ are computed from the current mini-batch
* $\gamma$ and $\beta$ are learnable per-channel parameters

---

## 3. What Exactly Is Being Normalized?

This is the central point.

For a 2D input:

$$
X \in \mathbb{R}^{B \times D}
$$

BatchNorm usually computes, for feature `j`:

$$
\mu_{B,j} = \frac{1}{B}\sum_{b=1}^{B}X_{b,j}
$$

$$
\sigma_{B,j}^2 = \frac{1}{B}\sum_{b=1}^{B}(X_{b,j}-\mu_{B,j})^2
$$

and outputs:

$$
Y_{b,j}
=
\gamma_j \frac{X_{b,j}-\mu_{B,j}}{\sqrt{\sigma_{B,j}^2+\epsilon}} + \beta_j
$$

So:

* each feature dimension is normalized separately
* statistics come from **multiple examples in the batch**

This is the opposite of LayerNorm:

* BatchNorm: normalize across **examples**
* LayerNorm: normalize across **features inside one example**

---

## 4. CNN Case: Why BatchNorm Fits Convolutions So Well

For CNNs, activations often have shape:

$$
X \in \mathbb{R}^{B \times C \times H \times W}
$$

For channel `c`, BatchNorm usually computes statistics over:

$$
(b,h,w)
$$

That means:

$$
\mu_{B,c}
=
\frac{1}{BHW}\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{w=1}^{W}X_{b,c,h,w}
$$

$$
\sigma_{B,c}^2
=
\frac{1}{BHW}\sum_{b,h,w}(X_{b,c,h,w}-\mu_{B,c})^2
$$

and then:

$$
Y_{b,c,h,w}
=
\gamma_c \frac{X_{b,c,h,w}-\mu_{B,c}}{\sqrt{\sigma_{B,c}^2+\epsilon}} + \beta_c
$$

So every spatial location in the same channel shares the same normalization statistics.

This is why BatchNorm is very natural in CNNs:

* channels represent feature maps
* examples and spatial locations provide many samples for stable statistics

---

## 5. Intuition

Suppose a batch has one feature channel with values:

$$
[2,\;4,\;6,\;8]
$$

Then:

$$
\mu_B = \frac{2+4+6+8}{4} = 5
$$

$$
\sigma_B^2 = \frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4} = 5
$$

So the normalized values are:

$$
\hat{x}
=
\frac{[2,4,6,8]-5}{\sqrt{5+\epsilon}}
\approx
[-1.34,\;-0.45,\;0.45,\;1.34]
$$

Interpretation:

* the batch mean becomes near `0`
* the batch variance becomes near `1`
* `\gamma` and `\beta` then learn the best final scale and shift

---

## 6. Training-Time vs Inference-Time Behavior

This is one of the most important BatchNorm details.

## 6.1 Training

At training time, BatchNorm uses the current mini-batch statistics:

$$
\mu_B,\;\sigma_B^2
$$

## 6.2 Inference

At inference time, you usually do **not** want predictions to depend on the current test batch.

So BatchNorm uses running estimates:

$$
\mu_{\text{running}},\;\sigma^2_{\text{running}}
$$

and computes:

$$
y = \gamma \frac{x-\mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}}+\epsilon}} + \beta
$$

These running statistics are updated during training, typically with exponential moving averages:

$$
\mu_{\text{running}}
\leftarrow
(1-\alpha)\mu_{\text{running}} + \alpha \mu_B
$$

$$
\sigma^2_{\text{running}}
\leftarrow
(1-\alpha)\sigma^2_{\text{running}} + \alpha \sigma_B^2
$$

So BatchNorm behaves differently in:

* `train()` mode
* `eval()` mode

That is a major practical difference from LayerNorm and RMSNorm.

---

## 7. Why BatchNorm Helps Optimization

BatchNorm helps because it keeps feature distributions more stable during training.

Historically, people often explain this as reducing **internal covariate shift**, but a more practical explanation is:

* it smooths optimization
* it stabilizes gradient scales
* it lets higher learning rates work more often
* it can act as a mild regularizer because batch statistics fluctuate

In CNNs, this often leads to:

* faster convergence
* better final accuracy

---

## 8. Why BatchNorm Is Less Natural for Transformers

BatchNorm depends on the current mini-batch.

That becomes awkward for sequence models because:

* batch size may be small
* sequence lengths vary
* autoregressive decoding often runs with batch size `1`
* training and inference behavior differ

Transformers usually prefer normalization methods that are **sample-local**, such as:

* LayerNorm
* RMSNorm

Those methods behave the same regardless of batch size.

That is why BatchNorm dominates CNNs far more than LLMs.

---

## 9. BatchNorm vs LayerNorm

Suppose:

$$
X \in \mathbb{R}^{B \times D}
$$

### BatchNorm

For feature `j`, compute:

$$
\mu_{B,j} = \frac{1}{B}\sum_{b=1}^{B}X_{b,j}
$$

So normalization is across examples.

### LayerNorm

For sample `b`, compute:

$$
\mu_b = \frac{1}{D}\sum_{j=1}^{D}X_{b,j}
$$

So normalization is across features.

The shortest contrast is:

> BatchNorm asks: "how does this feature behave across the batch?"  
> LayerNorm asks: "how should this one sample's vector be normalized?"

---

## 10. BatchNorm vs RMSNorm

BatchNorm:

* uses batch statistics
* centers and scales
* has different train/eval behavior

RMSNorm:

* uses only one sample's hidden vector
* scales but does not center
* same formula in train and eval

So these two are conceptually much farther apart than their names might suggest.

---

## 11. Practical PyTorch View

Common PyTorch modules:

```python
torch.nn.BatchNorm1d(num_features)
torch.nn.BatchNorm2d(num_features)
torch.nn.BatchNorm3d(num_features)
```

Typical meanings:

* `BatchNorm1d`: vectors or temporal features
* `BatchNorm2d`: image feature maps
* `BatchNorm3d`: volumetric or video-style feature maps

The key parameter is usually:

```python
num_features = C
```

meaning one learnable `gamma` and `beta` per channel.

---

## 12. Common Misunderstandings

### 12.1 "BatchNorm and LayerNorm are interchangeable"

No.

They normalize over different dimensions and have very different behavior.

### 12.2 "BatchNorm uses the same statistics at training and inference"

Wrong.

Training:

$$
\mu_B,\;\sigma_B^2
$$

Inference:

$$
\mu_{\text{running}},\;\sigma^2_{\text{running}}
$$

### 12.3 "BatchNorm only rescales activations"

No.

It also centers them by subtracting the batch mean.

---

## 13. Key Takeaway

The most important formula is:

$$
\operatorname{BatchNorm}(x)
=
\gamma \frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}} + \beta
$$

The most important interpretation is:

> BatchNorm normalizes one feature channel using statistics collected from the current mini-batch.

If you remember that, then the rest becomes easy to place:

* why it is so effective in CNNs
* why it needs running statistics
* why train and eval differ
* why LayerNorm / RMSNorm are usually preferred in Transformers
