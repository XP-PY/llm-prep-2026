# [GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) and Group Normalization

## 1. Why this note exists

`GroupNorm` is a normalization layer commonly used in CNN-style vision models, detection models, segmentation models, diffusion U-Nets, and robot vision backbones.

It is easy to confuse with:

* `BatchNorm`
* `LayerNorm`
* `InstanceNorm`

The clean mental model is:

> GroupNorm normalizes each sample independently, but splits channels into groups and computes statistics inside each group.

This sentence explains its most important property:

> unlike BatchNorm, GroupNorm does not depend on batch statistics.

That makes it useful when batch size is small or unstable.

---

## 2. Input Shape

For image-like activations, the input usually has shape:

$$
X \in \mathbb{R}^{B \times C \times H \times W}
$$

where:

* `B` = batch size
* `C` = number of channels
* `H, W` = spatial height and width

More generally, PyTorch's `GroupNorm` accepts:

$$
X \in \mathbb{R}^{N \times C \times *}
$$

where `*` means any number of extra spatial or sequence dimensions.

`GroupNorm` chooses a number of groups:

$$
G
$$

and requires:

$$
C \bmod G = 0
$$

Each group contains:

$$
C_g = \frac{C}{G}
$$

channels.

---

## 3. Core Formula

For each sample `n` and each group `g`, define the set of elements inside that group:

$$
\mathcal{S}_{n,g}
=
\left\{
X_{n,c,h,w}
\;\middle|\;
c \in \text{group}(g),\;
1 \le h \le H,\;
1 \le w \le W
\right\}
$$

The number of elements in one group is:

$$
m = C_gHW
$$

GroupNorm computes the mean for each sample and group:

$$
\mu_{n,g}
=
\frac{1}{m}
\sum_{c \in \text{group}(g)}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
X_{n,c,h,w}
$$

and the variance:

$$
\sigma_{n,g}^2
=
\frac{1}{m}
\sum_{c \in \text{group}(g)}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
\left(
X_{n,c,h,w} - \mu_{n,g}
\right)^2
$$

Then every element in that group is normalized:

$$
\hat{X}_{n,c,h,w}
=
\frac{
X_{n,c,h,w} - \mu_{n,g(c)}
}{
\sqrt{\sigma_{n,g(c)}^2 + \epsilon}
}
$$

where $g(c)$ is the group index containing channel `c`.

Finally, GroupNorm applies a learnable per-channel affine transform:

$$
Y_{n,c,h,w}
=
\gamma_c \hat{X}_{n,c,h,w} + \beta_c
$$

So the full formula is:

$$
\operatorname{GroupNorm}(X_{n,c,h,w})
=
\gamma_c
\frac{
X_{n,c,h,w} - \mu_{n,g(c)}
}{
\sqrt{\sigma_{n,g(c)}^2 + \epsilon}
}
+
\beta_c
$$

where:

* $\epsilon$ is a small numerical-stability constant
* $\gamma_c$ and $\beta_c$ are learnable per-channel parameters
* statistics are computed per sample and per group

---

## 4. What Exactly Is Being Normalized?

This is the central point.

For each sample `n`, GroupNorm splits the `C` channels into `G` groups.

For one group, it computes statistics over:

$$
\text{channels inside the group} \times \text{spatial positions}
$$

That means the statistics are computed over:

$$
(c,h,w)
$$

inside one sample.

So:

* different batch elements do **not** interact
* channels inside the same group interact
* spatial positions inside the same group interact
* channels from different groups do **not** share statistics

This is the main difference from BatchNorm.

BatchNorm computes statistics across the batch:

$$
(b,h,w)
$$

for each channel.

GroupNorm computes statistics inside each sample:

$$
(c,h,w)
$$

for each group.

---

## 5. Simple Numerical Example

Suppose:

$$
B=1,\quad C=4,\quad H=W=1,\quad G=2
$$

and one sample has channel values:

$$
x = [1,\;3,\;10,\;14]
$$

With `G = 2`, the channels are split into:

$$
\text{group 1} = [1,\;3]
$$

$$
\text{group 2} = [10,\;14]
$$

For group 1:

$$
\mu_1 = \frac{1+3}{2}=2
$$

$$
\sigma_1^2 = \frac{(1-2)^2+(3-2)^2}{2}=1
$$

So:

$$
\hat{x}_{1:2}
=
\frac{[1,3]-2}{\sqrt{1+\epsilon}}
\approx
[-1,\;1]
$$

For group 2:

$$
\mu_2 = \frac{10+14}{2}=12
$$

$$
\sigma_2^2 = \frac{(10-12)^2+(14-12)^2}{2}=4
$$

So:

$$
\hat{x}_{3:4}
=
\frac{[10,14]-12}{\sqrt{4+\epsilon}}
\approx
[-1,\;1]
$$

The final output is then scaled and shifted per channel:

$$
y_c = \gamma_c \hat{x}_c + \beta_c
$$

---

## 6. Relation to BatchNorm, LayerNorm, and InstanceNorm

GroupNorm can be viewed as a bridge between LayerNorm and InstanceNorm.

For image-like input:

$$
X \in \mathbb{R}^{B \times C \times H \times W}
$$

### 6.1 `G = 1`: LayerNorm-like behavior

If:

$$
G = 1
$$

then all channels belong to one group.

The statistics are computed over:

$$
C \times H \times W
$$

inside each sample.

This is similar to LayerNorm over `[C, H, W]`, although the affine parameterization in PyTorch differs from a fully general `LayerNorm([C, H, W])`.

### 6.2 `G = C`: InstanceNorm-like behavior

If:

$$
G = C
$$

then each channel is its own group.

The statistics are computed over:

$$
H \times W
$$

inside each sample and channel.

This is similar to InstanceNorm without running statistics.

### 6.3 Intermediate `G`: actual GroupNorm

The common case is:

$$
1 < G < C
$$

For example:

$$
G = 32
$$

This means nearby channels in the same group share statistics, but the entire channel dimension is not collapsed into one global normalization group.

---

## 7. Why GroupNorm Helps with Small Batches

BatchNorm needs batch statistics:

$$
\mu_{B,c},\quad \sigma_{B,c}^2
$$

If the batch is small, these estimates can be noisy.

This often happens in:

* object detection
* segmentation
* video models
* diffusion U-Nets
* robotics models with large images
* distributed training with small per-GPU batch size

GroupNorm avoids this problem because its statistics are computed inside each sample:

$$
\mu_{n,g},\quad \sigma_{n,g}^2
$$

So training behavior is much less sensitive to batch size.

Another practical advantage:

> GroupNorm behaves the same way during training and inference.

There are no running mean or running variance buffers like BatchNorm.

---

## 8. Training-Time vs Inference-Time Behavior

BatchNorm has different training and inference behavior:

* training uses current mini-batch statistics
* inference uses running statistics

GroupNorm does not.

GroupNorm always computes statistics from the current sample:

$$
\mu_{n,g},\quad \sigma_{n,g}^2
$$

Therefore:

* no running mean
* no running variance
* no dependency on test batch composition
* less mismatch between training and inference

This is a major reason GroupNorm is popular in vision tasks with small or variable batch sizes.

---

## 9. Choosing the Number of Groups

The only hard constraint is:

$$
C \bmod G = 0
$$

Common choices:

* `num_groups = 32` when the channel count is large enough
* `num_groups = 16` or `8` for smaller models
* `num_groups = 1` for LayerNorm-like behavior on convolutional features
* `num_groups = C` for InstanceNorm-like behavior

A useful practical heuristic is:

> keep enough channels per group so the statistics are stable, but do not put all channels into one group unless that is intentional.

If each group is too small, the estimated variance may be noisy.

If each group is too large, GroupNorm becomes closer to LayerNorm over the whole feature map, which may remove too much structure.

---

## 10. PyTorch Usage

Basic usage:

```python
import torch
from torch import nn

x = torch.randn(8, 64, 32, 32)

norm = nn.GroupNorm(
    num_groups=8,
    num_channels=64,
)

y = norm(x)
print(y.shape)  # torch.Size([8, 64, 32, 32])
```

Common variants:

```python
# Standard GroupNorm: 8 groups, 64 channels.
nn.GroupNorm(num_groups=8, num_channels=64)

# LayerNorm-like for convolutional features.
nn.GroupNorm(num_groups=1, num_channels=64)

# InstanceNorm-like behavior.
nn.GroupNorm(num_groups=64, num_channels=64)
```

Important PyTorch detail:

```python
nn.GroupNorm(num_groups=G, num_channels=C)
```

expects channel-first input:

```text
[N, C, *]
```

For image tensors, this is usually:

```text
[B, C, H, W]
```

---

## 11. GroupNorm in Transformers

GroupNorm is much less common than LayerNorm or RMSNorm in Transformers.

The reason is shape and semantics.

Transformer hidden states usually have shape:

$$
X \in \mathbb{R}^{B \times T \times D}
$$

LayerNorm normalizes each token independently over `D`.

GroupNorm expects channel-first input, so one might transpose to:

$$
X' \in \mathbb{R}^{B \times D \times T}
$$

Then GroupNorm statistics are computed over grouped hidden channels and the time dimension `T`.

That means different token positions in the same sequence can interact through normalization statistics.

For autoregressive language modeling, this is usually undesirable because future tokens may affect the normalization of earlier tokens during training.

So:

* use `LayerNorm` or `RMSNorm` for standard Transformers
* use `GroupNorm` mainly for convolutional or vision-style backbones
* be careful if applying GroupNorm to sequence tensors

---

## 12. Comparison Table

| Method | Statistics computed over | Batch-dependent? | Running stats? | Common use |
|---|---:|---:|---:|---|
| BatchNorm | batch + spatial, per channel | Yes | Yes | CNN classification with large batches |
| GroupNorm | channel group + spatial, per sample | No | No | detection, segmentation, diffusion U-Nets |
| LayerNorm | feature dimension, per sample/token | No | No | Transformers |
| RMSNorm | feature RMS, per sample/token | No | No | modern LLMs |
| InstanceNorm | spatial dimensions, per sample/channel | No | usually no | style transfer, image generation |

---

## 13. Common Mistakes

Mistake 1:

> Thinking GroupNorm uses batch statistics.

It does not. GroupNorm is batch-size independent.

Mistake 2:

> Setting `num_groups` to a value that does not divide `num_channels`.

PyTorch requires:

$$
C \bmod G = 0
$$

Mistake 3:

> Treating `GroupNorm(num_groups=1)` as exactly identical to every LayerNorm configuration.

It is only LayerNorm-like if the normalized dimensions match. PyTorch's affine parameters are per channel for GroupNorm, while LayerNorm can have per-element affine parameters over the full normalized shape.

Mistake 4:

> Applying GroupNorm to autoregressive sequence features without checking whether token positions are mixed.

If the time dimension is included in the normalized axes, future positions can influence earlier positions during training.

---

## 14. Key Takeaways

* GroupNorm normalizes each sample independently.
* Channels are split into groups, and each group has its own mean and variance.
* It does not use running statistics.
* It behaves the same during training and inference.
* It is more robust than BatchNorm when batch size is small.
* `G = 1` is LayerNorm-like over convolutional features.
* `G = C` is InstanceNorm-like.
* GroupNorm is mostly a vision / convolutional backbone normalization layer, while LayerNorm and RMSNorm remain the standard choices for Transformers.

