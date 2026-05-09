# [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) and Binary Logistic Loss

## 1. Why this note exists

`sigmoid` appears in many places that are easy to mix together:

* as an activation function
* as a probability mapping
* inside binary cross-entropy
* inside pairwise matching losses such as [SigLIP](../Large_Models/SigLIP.md)

The clean mental model is:

> sigmoid turns one real-valued logit into one independent probability.

That "independent" part is the key difference from `softmax`.

---

## 2. Definition

For a scalar logit $x$:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Its output range is:

$$
\sigma(x) \in (0, 1)
$$

So sigmoid is often used when we want to interpret one score as:

* probability of a binary label
* confidence that one pair matches
* gate strength in a neural network

---

## 3. Shape intuition

Sigmoid has an S-shaped curve:

* large negative input -> output near `0`
* input near `0` -> output near `0.5`
* large positive input -> output near `1`

This means:

* sign decides which side of `0.5` we are on
* magnitude decides confidence

Examples:

$$
\sigma(0) = 0.5, \quad
\sigma(2) \approx 0.88, \quad
\sigma(-2) \approx 0.12
$$

---

## 4. Derivative

The derivative is:

$$
\sigma'(x) = \sigma(x)\left(1 - \sigma(x)\right)
$$

This is useful because:

* it is largest around `x = 0`
* it becomes very small when `x` is very positive or very negative

So sigmoid can saturate:

* if the logit is too large, gradients become tiny
* if the logit is too negative, gradients also become tiny

This is one reason modern hidden layers often prefer ReLU, GeLU, or SwiGLU instead of stacking many sigmoids.

---

## 5. Sigmoid vs softmax

This distinction matters a lot.

## 5.1 Sigmoid

Sigmoid treats each score independently:

$$
p_i = \sigma(x_i)
$$

So multiple outputs can all be high at the same time.

This is natural for:

* binary classification
* multi-label classification
* pairwise matching

## 5.2 Softmax

Softmax couples all scores together:

$$
p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

So increasing one score changes the probabilities of all others.

This is natural for:

* one-of-K classification
* row-wise ranking across candidates
* contrastive objectives such as CLIP

The shortest contrast is:

> sigmoid means "is this item positive?"
> softmax means "which item wins among these candidates?"

---

## 6. [Binary cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

If the target is $y \in \{0, 1\}$ and the predicted probability is $p = \sigma(x)$, then binary cross-entropy is:

$$
\mathcal{L}_{\text{BCE}} = -\left[y \log p + (1-y)\log(1-p)\right]
$$

Substituting $p = \sigma(x)$ gives logistic loss.

Interpretation:

* if `y = 1`, push `p` toward `1`
* if `y = 0`, push `p` toward `0`

---

## 7. Why [`BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) is preferred

In PyTorch, you will often see:

```python
torch.nn.BCEWithLogitsLoss()
```

instead of:

```python
torch.sigmoid(logits)
loss = torch.nn.BCELoss()(probs, targets)
```

Why?

Because `BCEWithLogitsLoss` combines:

1. sigmoid
2. binary cross-entropy
3. numerically stable implementation

into one operation.

This avoids instability from explicitly computing:

* `log(sigmoid(x))`
* `log(1 - sigmoid(x))`

for very large positive or negative logits.

In practice:

* use `sigmoid` when you want probabilities for inspection or inference
* use `BCEWithLogitsLoss` when you are training from raw logits

---

## 8. Relation to [`softplus`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.Softplus.html)

Binary logistic loss can be rewritten in a numerically stable form.

There are two common label encodings:

* $y \in \{0, 1\}$
* $z \in \{-1, +1\}$ with $z = 2y - 1$

Starting from BCE with $y \in \{0,1\}$:

$$
\mathcal{L}_{\text{BCE}} = -\left[y \log \sigma(x) + (1-y)\log(1-\sigma(x))\right]
$$

we can rewrite the same loss with $z \in \{-1, +1\}$.

Let $x$ be the logit. Then:

$$
\mathcal{L} = -\log \sigma(zx) = \operatorname{softplus}(-zx)
$$

This is why some formulas seem to place the label outside the `log`, while others place it inside:

* with $y \in \{0,1\}$, the label selects between the positive and negative BCE terms
* with $z \in \{-1,+1\}$, the label flips the sign of the logit inside one unified expression

The correspondence is:

$$
\begin{aligned}
y = 1 &\Longleftrightarrow z = +1 \Longrightarrow -\log \sigma(x) \\
y = 0 &\Longleftrightarrow z = -1 \Longrightarrow -\log \sigma(-x)
\end{aligned}
$$

and since:

$$
\sigma(-x) = 1 - \sigma(x)
$$

the two forms are mathematically equivalent.

where:

$$
\operatorname{softplus}(u) = \log(1 + e^u)
$$

This is why many implementations do not explicitly call `sigmoid` inside the training loss even though the loss is conceptually a sigmoid loss.

---
