# AdamW Optimizer

## Principle: Why AdamW Matters for Large Model Training

[Adam (Adaptive Moment Estimation)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)) is popular because it adapts learning rates per parameter using momentum (first moment) and RMS of gradients (second moment), smoothing updates in noisy or sparse gradients common in LLMs.

However, **standard Adam couples weight decay (L2 regularization) with the adaptive step**, causing issues at scale: parameters with small gradients get overly regularized, hurting generalization in billion-parameter Transformers.
AdamW decouples weight decay, applying it separately from the adaptive update. This improves stability, generalization, and is the default in LLaMA, GPT-NeoX, and most modern LLMs.

## Detailed Derivation

Start with [**Adam**](https://arxiv.org/abs/1412.6980):

1. Oringal Gradient: $g_t = \nabla \mathcal{L}(\theta_{t-1})$
2. Incorporate weight decay as an L2 regularizer on the loss: $g_t = \nabla \mathcal{L}(\theta_{t-1}) + \lambda \theta_{t-1}$
3. First moment (momentum): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
4. Second moment (adaptive scaling): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
5. Parameter update (coupled Adam):
   $$
   \theta_t = \theta_{t-1} - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} \right)
   $$

**The key issue**:
Because the weight decay term $\lambda \theta_{t-1}$ is incorporated into $g_t$, it becomes subject to the same adaptive scaling factor $1 / (\sqrt{v_t} + \epsilon)$ as the gradient itself. In practice, this means:
    - When the gradient is large ($v_t$ is large), the effective weight decay is suppressed.
    - When the gradient is small ($v_t$ is small), the effective weight decay is amplified, potentially becoming excessively large.

This coupling between weight decay and adaptive learning rates undermines the intended regularization effect, leading to poor generalization, especially in large-scale models like Transformers.

[**AdamW**](https://arxiv.org/abs/1711.05101) fix:

1. **Clean moments (same as original Adam)**

$$
g_t = \nabla \mathcal{L}(\theta_{t-1})
$$

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

2. **Separate decoupled weight decay**:

$$
\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1}
$$

3. **Adaptive step**:

$$
\theta_t = \theta_{t} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

→ Decay strength is **exactly** `ηλ` for every parameter — independent of gradient statistics.

## Step-by-Step Code Implementation

[Python script](../../src/part1_optimizers.ipynb)
