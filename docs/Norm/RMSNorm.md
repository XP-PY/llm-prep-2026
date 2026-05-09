# [RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html) and Root Mean Square Normalization

## 1. Why this note exists

`RMSNorm` appears in many modern LLMs such as LLaMA, Gemma, Falcon, and GPT-NeoX style architectures.

It is often described loosely as:

> "LayerNorm without mean subtraction."

That summary is directionally correct, but incomplete.

The important questions are:

* what is the exact formula?
* what does it normalize and what does it not normalize?
* why can we remove centering and still train large models well?
* how is it different from `LayerNorm` in practice?

The clean mental model is:

> RMSNorm normalizes **the magnitude** of a hidden vector, but does **not** force its mean to be zero.

---

## 2. Core Formula

For one hidden vector:

$$
x = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d
$$

RMSNorm first computes the root mean square:

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}
$$

Then it rescales the vector:

$$
\hat{x}_i = \frac{x_i}{\operatorname{RMS}(x)}
$$

Finally it applies a learnable element-wise scale:

$$
y_i = \gamma_i \hat{x}_i
$$

So the full formula is:

$$
\operatorname{RMSNorm}(x)
=
\gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$

where:

* `d` is the hidden dimension
* `\epsilon` is for numerical stability
* `\gamma \in \mathbb{R}^d` is a learnable scale vector

Unlike LayerNorm, there is usually:

* no mean subtraction
* no `\beta` bias term

---

## 3. What Exactly Is Being Normalized?

For a Transformer hidden state:

$$
X \in \mathbb{R}^{B \times T \times D}
$$

RMSNorm is usually applied over the last dimension `D`.

So for each token `(b,t)`:

$$
\operatorname{RMS}_{b,t}
=
\sqrt{\frac{1}{D}\sum_{j=1}^{D}X_{b,t,j}^2 + \epsilon}
$$

and the output is:

$$
Y_{b,t,j}
=
\gamma_j \frac{X_{b,t,j}}{\operatorname{RMS}_{b,t}}
$$

So just like LayerNorm:

* different batch elements do **not** interact
* different time steps do **not** interact
* normalization happens only inside one token vector

The difference is not **which dimension** is normalized.  
The difference is **what statistic** is used.

---

## 4. RMSNorm vs LayerNorm

LayerNorm uses:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i,
\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

and outputs:

$$
\operatorname{LayerNorm}(x)
=
\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
$$

RMSNorm uses:

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

and outputs:

$$
\operatorname{RMSNorm}(x)
=
\gamma \odot \frac{x}{\operatorname{RMS}(x)}
$$

So:

* LayerNorm normalizes **mean and scale**
* RMSNorm normalizes **scale only**

This is the most important conceptual distinction.

---

## 5. Intuition

Suppose:

$$
x = [2,\;4,\;6]
$$

Then:

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{2^2+4^2+6^2}{3}}
=
\sqrt{\frac{56}{3}}
\approx 4.320
$$

So the normalized vector is:

$$
\hat{x}
=
\frac{1}{4.320}[2,4,6]
\approx
[0.463,\;0.926,\;1.389]
$$

Notice what happened:

* the overall magnitude was normalized
* but the vector was **not centered**

Its mean is still positive.

That is exactly the point:

> RMSNorm keeps directional / offset information that LayerNorm would partially remove.

---

## 6. Why Removing the Mean Can Still Work

This is the main practical question.

LayerNorm assumes that both:

* offset
* scale

should be normalized away.

RMSNorm assumes that, in many deep Transformer stacks, the most important source of instability is **magnitude**, not necessarily mean shift.

So instead of forcing:

$$
\mathbb{E}[x] = 0
$$

RMSNorm only controls:

$$
\frac{1}{d}\sum_{i=1}^{d}x_i^2
$$

The practical intuition is:

* residual streams already carry structured information
* subtracting the mean may be unnecessary
* controlling the activation norm is often enough for stable optimization

This is why RMSNorm often matches or improves LayerNorm in large LLMs while being simpler.

---

## 7. Important Properties

## 7.1 Scale Invariance

If:

$$
x' = ax
$$

for positive scalar `a`, then:

$$
\operatorname{RMSNorm}(x') \approx \operatorname{RMSNorm}(x)
$$

because both numerator and denominator scale by `a`.

So RMSNorm is robust to overall activation magnitude.

## 7.2 Not Shift Invariant

If:

$$
x' = x + c\mathbf{1}
$$

then in general:

$$
\operatorname{RMSNorm}(x') \neq \operatorname{RMSNorm}(x)
$$

because RMSNorm does **not** subtract the mean.

This is the clearest mathematical difference from LayerNorm:

* LayerNorm is roughly shift-invariant
* RMSNorm is not

---

## 8. Why Modern LLMs Like RMSNorm

There are three common reasons.

### 8.1 Simpler computation

LayerNorm needs:

* mean
* variance

RMSNorm needs only:

* mean square

So it is slightly cheaper.

### 8.2 Cleaner residual stream

In decoder-only LLMs, the residual path is very important.  
RMSNorm only rescales the residual vector, instead of also re-centering it.

That can be a better fit for deep residual architectures.

### 8.3 Strong empirical results

In practice, many large language models train very well with RMSNorm, so the simpler formulation becomes attractive.

This is one of those cases where:

> a slightly weaker mathematical constraint can be a better engineering choice.

---

## 9. RMSNorm in Transformers

In Pre-Norm LLMs, RMSNorm often appears as:

$$
h_{\ell+1} = h_\ell + F_\ell(\operatorname{RMSNorm}(h_\ell))
$$

where `F_\ell` is:

* attention
* FFN

So RMSNorm stabilizes the input to each sublayer while keeping the residual path direct.

This is the common pattern in modern decoder-only architectures.

---

## 10. Relation to Hidden-State Norm

If we define:

$$
\|x\|_2 = \sqrt{\sum_{i=1}^{d}x_i^2}
$$

then:

$$
\operatorname{RMS}(x) = \frac{\|x\|_2}{\sqrt{d}}
$$

So RMSNorm can also be written as:

$$
\operatorname{RMSNorm}(x)
=
\gamma \odot \frac{\sqrt{d}\,x}{\|x\|_2}
$$

ignoring `\epsilon`.

This makes the geometry clearer:

> RMSNorm mostly fixes vector length, not vector mean.

---

## 11. Practical PyTorch View

In PyTorch:

```python
torch.nn.RMSNorm(D, eps=1e-6)
```

For:

```python
x.shape == [B, T, D]
```

this typically means:

* normalize over the last dimension `D`
* keep one learnable scale vector of shape `[D]`

There is no explicit `beta` term by default in the usual RMSNorm formulation.

---

## 12. RMSNorm vs LayerNorm vs BatchNorm

### 12.1 RMSNorm vs LayerNorm

RMSNorm:

$$
\frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}}
$$

LayerNorm:

$$
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

Difference:

* RMSNorm controls scale only
* LayerNorm controls both centering and scale

### 12.2 RMSNorm vs BatchNorm

BatchNorm normalizes across the batch dimension and often also spatial dimensions.

RMSNorm normalizes inside one sample / one token vector.

So RMSNorm is much more natural for:

* autoregressive decoding
* variable-length sequences
* batch size `1`

---

## 13. Common Misunderstandings

### 13.1 "RMSNorm is just a faster LayerNorm"

Not exactly.

It is cheaper, but also mathematically different:

* no centering
* no `\beta`
* not shift-invariant

### 13.2 "RMSNorm and LayerNorm normalize the same statistic"

Wrong.

LayerNorm uses:

$$
\mu,\;\sigma^2
$$

RMSNorm uses:

$$
\frac{1}{d}\sum x_i^2
$$

### 13.3 "If there is no mean subtraction, normalization is incomplete"

Not necessarily.

For modern deep LLMs, controlling magnitude is often enough to get the stability we need.

---

## 14. Key Takeaway

The most important formula is:

$$
\operatorname{RMSNorm}(x)
=
\gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$

The most important interpretation is:

> RMSNorm normalizes the **size** of a hidden vector, but does not force it to have zero mean.

If you remember that, then the rest becomes much easier:

* why it differs from LayerNorm
* why it is popular in modern LLMs
* why it is cheaper
* why it still works well in deep Transformer residual streams
