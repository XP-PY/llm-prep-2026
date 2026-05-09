# [GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) and Gaussian Error Linear Unit

## 1. Why this note exists

`GELU` is one of the most important activation functions in modern deep learning, especially in:

* BERT
* Vision Transformers
* many encoder-style Transformers

It often gets described loosely as:

> a smoother alternative to ReLU

That is true, but not enough.

The useful questions are:

* what is the exact formula?
* why does a Gaussian CDF appear?
* how is it different from ReLU and SiLU?
* why does it work so well in Transformers?

The clean mental model is:

> GELU softly gates an input by how likely a standard Gaussian variable would keep it.

---

## 2. Core Formula

The original GELU definition is:

$$
\operatorname{GELU}(x) = x \, \Phi(x)
$$

where:

$$
\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2}\,dt
$$

is the cumulative distribution function (CDF) of a standard Gaussian.

So GELU multiplies the input `x` by a smooth gate `\Phi(x) \in (0,1)`.

That means:

* large negative `x` -> output near `0`
* large positive `x` -> output near `x`
* values near `0` are only partially passed through

---

## 3. Common Approximation

The exact Gaussian CDF is somewhat expensive, so implementations often use an approximation.

A common one is:

$$
\operatorname{GELU}(x)
\approx
\frac{1}{2}x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
$$

PyTorch also supports this approximate version.

So in practice, when people say "GELU", they often mean:

* exact GELU in theory
* tanh-approx GELU in implementation

---

## 4. Intuition

ReLU makes a hard decision:

$$
\operatorname{ReLU}(x) = \max(0, x)
$$

Either:

* zero the value
* keep it fully

GELU is softer:

$$
\operatorname{GELU}(x) = x \Phi(x)
$$

So it behaves like a **probabilistic gate**:

* very negative values are mostly suppressed
* very positive values are mostly preserved
* moderately positive or moderately negative values are only partially kept

This is why GELU is often described as:

> smoother than ReLU, less abrupt around zero, and more graded in how it keeps information.

---

## 5. Example Values

Useful reference points:

$$
\operatorname{GELU}(0) = 0
$$

because:

$$
\Phi(0) = 0.5
\quad \Rightarrow \quad
0 \cdot 0.5 = 0
$$

Other rough values:

$$
\operatorname{GELU}(1) \approx 0.84
$$

$$
\operatorname{GELU}(-1) \approx -0.16
$$

This is an important difference from ReLU:

* ReLU would output `0` for `-1`
* GELU outputs a small negative value instead

So GELU is **not** a strict nonnegative activation.

---

## 6. Why the Gaussian CDF Appears

The original paper motivates GELU as stochastic regularization intuition.

You can view:

$$
\Phi(x)
$$

as the probability of keeping the input, where larger `x` means larger keep-probability.

So:

$$
\operatorname{GELU}(x) = x \Phi(x)
$$

can be interpreted as:

> keep the input in proportion to how confident the gate is that it should pass.

This is a probabilistic, input-dependent soft gating rule.

That is more nuanced than ReLU's hard threshold at zero.

---

## 7. Derivative

Starting from:

$$
\operatorname{GELU}(x) = x\Phi(x)
$$

the derivative is:

$$
\frac{d}{dx}\operatorname{GELU}(x)
=
\Phi(x) + x\phi(x)
$$

where:

$$
\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}
$$

is the standard Gaussian density.

So:

$$
\operatorname{GELU}'(x) = \Phi(x) + x\phi(x)
$$

This derivative is smooth everywhere.

That is one reason optimization with GELU often behaves more gently than with ReLU.

---

## 8. Shape of the Function

GELU is smooth and curved around zero.

Key behaviors:

* for large positive `x`, `\Phi(x) \to 1`, so:

$$
\operatorname{GELU}(x) \approx x
$$

* for large negative `x`, `\Phi(x) \to 0`, so:

$$
\operatorname{GELU}(x) \approx 0
$$

but it approaches zero smoothly from the negative side rather than sharply clipping.

This gives GELU a shape that is:

* smoother than ReLU
* less symmetric than `tanh`
* less aggressively gated than hard thresholding

---

## 9. GELU vs ReLU

### 9.1 ReLU

$$
\operatorname{ReLU}(x) = \max(0,x)
$$

Properties:

* simple
* cheap
* sparse activations
* zero gradient on the negative side

### 9.2 GELU

$$
\operatorname{GELU}(x) = x\Phi(x)
$$

Properties:

* smooth
* nonzero values for some negative inputs
* soft gating instead of hard cutoff

The shortest contrast is:

> ReLU asks "is this positive?"  
> GELU asks "how strongly should this be passed through?"

---

## 10. GELU vs SiLU / Swish

SiLU is:

$$
\operatorname{SiLU}(x) = x \sigma(x)
$$

GELU is:

$$
\operatorname{GELU}(x) = x \Phi(x)
$$

So both have the same general structure:

$$
x \times \text{smooth gate}(x)
$$

Difference:

* SiLU uses `sigmoid`
* GELU uses Gaussian CDF

So both are smooth self-gated activations, but they use different gate shapes.

In practice:

* GELU is very common in encoder Transformers
* SiLU / Swish appears a lot in decoder LLMs and gated FFNs

---

## 11. Why GELU Works Well in Transformers

Transformers need activations that behave well under:

* deep residual stacks
* large hidden dimensions
* subtle optimization dynamics

GELU helps because:

* it is smooth
* it does not abruptly kill slightly negative activations
* it gives a graded transition around zero

In FFNs, this often gives better behavior than simple ReLU.

Historically, that is one reason BERT and many Transformer encoder models adopted GELU.

---

## 12. GELU in Feed-Forward Networks

A standard Transformer FFN often looks like:

$$
\operatorname{FFN}(x) = W_2 \, \operatorname{GELU}(W_1x + b_1) + b_2
$$

So GELU sits between:

* the up-projection
* the down-projection

Its role is to introduce nonlinearity while preserving smooth optimization.

---

## 13. Practical PyTorch View

In PyTorch:

```python
torch.nn.GELU()
```

or:

```python
torch.nn.GELU(approximate="tanh")
```

The second version is often used for speed.

So conceptually:

* exact GELU = theory definition
* tanh-GELU = practical approximation

---

## 14. Common Misunderstandings

### 14.1 "GELU is just a smoothed ReLU"

Only partially true.

It is smoother, but its real definition is:

$$
x\Phi(x)
$$

which is a Gaussian-CDF gate, not just an arbitrary smoothing of `max(0,x)`.

### 14.2 "GELU outputs only nonnegative values"

Wrong.

For negative inputs, GELU can output small negative values.

For example:

$$
\operatorname{GELU}(-1) \approx -0.16
$$

### 14.3 "GELU and SiLU are the same"

No.

They are similar in form, but use different gates:

$$
\operatorname{GELU}(x) = x\Phi(x)
$$

$$
\operatorname{SiLU}(x) = x\sigma(x)
$$

---

## 15. Key Takeaway

The most important formula is:

$$
\operatorname{GELU}(x) = x\Phi(x)
$$

The most important interpretation is:

> GELU softly passes an input according to a smooth Gaussian-CDF gate.

If you remember that, then the rest becomes easy to place:

* why it differs from ReLU
* why it is close in spirit to SiLU
* why it works well in Transformer FFNs
* why modern implementations often use the tanh approximation
