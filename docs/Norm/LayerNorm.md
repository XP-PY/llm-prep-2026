# [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and Layer Normalization

## 1. Why this note exists

`LayerNorm` is one of the most common operations in Transformers, but it is also easy to misunderstand:

* over **which dimension** does it normalize?
* why does it need both `gamma` and `beta`?
* why does it work well in sequence models?
* why do modern LLMs often replace it with `RMSNorm`?

The clean mental model is:

> LayerNorm normalizes **each token's hidden vector independently** across its feature dimension.

That one sentence already explains why it behaves very differently from `BatchNorm`.

---

## 2. Core Formula

For one hidden vector:

$$
x = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d
$$

LayerNorm first computes the mean:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$

and the variance:

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

Then it standardizes each component:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Finally it applies a learnable affine transform:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

So the full LayerNorm formula is:

$$
\operatorname{LayerNorm}(x)
=
\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:

* $d$ is the hidden dimension
* $\epsilon$ is a small constant for numerical stability
* $\gamma, \beta \in \mathbb{R}^d$ are learnable parameters
* $\odot$ means element-wise multiplication

---

## 3. What Exactly Is Being Normalized?

This is the most important point.

For a Transformer hidden state:

$$
X \in \mathbb{R}^{B \times T \times D}
$$

where:

* `B` = batch size
* `T` = sequence length
* `D` = hidden size

LayerNorm is usually applied over the **last dimension** `D`.

That means for each token `(b, t)`, we compute:

$$
\mu_{b,t} = \frac{1}{D}\sum_{j=1}^{D} X_{b,t,j}
$$

$$
\sigma^2_{b,t} = \frac{1}{D}\sum_{j=1}^{D}(X_{b,t,j} - \mu_{b,t})^2
$$

and output:

$$
Y_{b,t,j}
=
\gamma_j \frac{X_{b,t,j} - \mu_{b,t}}{\sqrt{\sigma^2_{b,t} + \epsilon}} + \beta_j
$$

So:

* different samples in the batch do **not** interact
* different time steps do **not** interact
* only the **features inside one token vector** interact

This is why LayerNorm is natural for NLP and Transformers.

---

## 4. Intuition

Suppose one token's hidden vector is:

$$
x = [2,\; 4,\; 6]
$$

Then:

$$
\mu = \frac{2+4+6}{3} = 4
$$

$$
\sigma^2 = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{8}{3}
$$

So the normalized vector is:

$$
\hat{x}
=
\frac{[2,4,6]-4}{\sqrt{8/3+\epsilon}}
\approx
[-1.225,\;0,\;1.225]
$$

Interpretation:

* subtracting $\mu$ removes the **overall offset**
* dividing by $\sqrt{\sigma^2 + \epsilon}$ removes the **overall scale**
* $\gamma$ and $\beta$ let the model learn the best output scale and shift again

So LayerNorm does **not** destroy information completely.  
It removes a raw, unstable parameterization and then lets the model learn a better one.

---

## 5. Why `gamma` and `beta` Are Necessary

If we only standardized:

$$
\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

then every token would always have:

* mean `0`
* variance about `1`

That may be too restrictive.

So LayerNorm adds:

$$
y = \gamma \odot \hat{x} + \beta
$$

This gives the network freedom to learn:

* which features should be amplified
* which should be suppressed
* what output scale works best for the next sublayer

You can think of it as:

> normalize first for stability, then re-parameterize for expressiveness.

---

## 6. Important Properties

Ignoring $\epsilon$ for intuition:

### 6.1 Shift Invariance

If we add the same constant `c` to every component:

$$
x' = x + c\mathbf{1}
$$

then:

$$
\operatorname{LN}(x') \approx \operatorname{LN}(x)
$$

because mean subtraction removes the common offset.

### 6.2 Scale Invariance

If we multiply by a positive scalar `a`:

$$
x' = ax
$$

then:

$$
\operatorname{LN}(x') \approx \operatorname{LN}(x)
$$

because both numerator and denominator scale by `a`.

This helps make optimization more stable: the network becomes less sensitive to raw activation magnitude.

---

## 7. LayerNorm vs BatchNorm

These two are often confused.

## 7.1 BatchNorm

BatchNorm normalizes using statistics across the **batch**:

$$
\mu_{\text{batch}} = \frac{1}{m}\sum_{k=1}^{m} x_k
$$

So one sample depends on the other samples in the mini-batch.

That works well in CNNs, but has drawbacks in sequence models:

* batch size may vary
* sequence lengths vary
* autoregressive decoding often uses batch size `1`

## 7.2 LayerNorm

LayerNorm normalizes **within one sample / one token**:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

So it works the same:

* at training time
* at inference time
* with large or small batches
* even with batch size `1`

This is one major reason Transformers prefer LayerNorm.

The shortest contrast is:

> BatchNorm normalizes across examples.  
> LayerNorm normalizes across features.

---

## 8. LayerNorm in Transformers

LayerNorm is usually placed around attention and FFN sublayers.

### 8.1 Post-Norm Transformer

Original Transformer-style:

$$
h_{\ell+1} = \operatorname{LN}\left(h_\ell + F_\ell(h_\ell)\right)
$$

where $F_\ell$ is attention or FFN.

### 8.2 Pre-Norm Transformer

Modern LLM-style:

$$
h_{\ell+1} = h_\ell + F_\ell(\operatorname{LN}(h_\ell))
$$

This is usually more stable for deep models because the residual path stays cleaner.

That is why many modern decoder-only LLMs use **Pre-Norm**.

---

## 9. Why LayerNorm Helps Optimization

LayerNorm helps because it keeps hidden-state statistics more controlled.

More concretely, it reduces:

* exploding hidden magnitudes
* sensitivity to parameter scale
* unstable depth-wise signal propagation

You can think of it as making each token vector live in a more predictable range before it enters attention or FFN.

This usually improves:

* training stability
* gradient flow
* robustness to hyperparameter choices

It is not magic, but it makes optimization much easier.

---

## 10. Why Modern LLMs Often Use RMSNorm Instead

LayerNorm uses both:

* mean subtraction
* variance normalization

RMSNorm removes the centering step and keeps only root-mean-square scaling:

$$
\operatorname{RMSNorm}(x)
=
\gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$

So compared with LayerNorm:

* no $\mu$
* no subtraction of the mean
* usually no $\beta$

Why do this?

* slightly cheaper
* simpler
* works very well in large LLMs

But conceptually, RMSNorm is easier to understand after you understand LayerNorm first.

---

## 11. Practical PyTorch View

In PyTorch:

```python
torch.nn.LayerNorm(normalized_shape=D, eps=1e-5)
```

For a tensor:

```python
x.shape == [B, T, D]
```

this usually means:

* normalize over the last dimension `D`
* use one learnable `gamma` and `beta`, each of shape `[D]`

So the operation is token-wise over hidden features.

---

## 12. Common Misunderstandings

### 12.1 "LayerNorm uses batch statistics"

Wrong. That is BatchNorm, not LayerNorm.

### 12.2 "LayerNorm forces all outputs to stay standardized forever"

Not exactly. The normalization step standardizes the vector, but the learnable affine parameters:

$$
\gamma, \beta
$$

let the model reshape the output distribution again.

### 12.3 "LayerNorm is the same as RMSNorm"

No.

LayerNorm:

$$
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

RMSNorm:

$$
\frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}}
$$

RMSNorm keeps scale normalization but removes centering.

---

## 13. Key Takeaway

The most important formula is:

$$
\operatorname{LayerNorm}(x)
=
\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

The most important interpretation is:

> for each token, LayerNorm normalizes across the hidden dimension, not across the batch.

If you remember that, then the rest becomes much easier:

* why it works in Transformers
* why train and inference use the same rule
* why it differs from BatchNorm
* why RMSNorm is a simplification rather than a completely different idea
