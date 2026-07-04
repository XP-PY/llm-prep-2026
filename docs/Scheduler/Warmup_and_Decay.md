# Warmup and Decay Learning Rate Schedulers

## Convenient Links

* [PyTorch `LinearLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)
* [PyTorch `CosineAnnealingLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
* [PyTorch `SequentialLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html)

## 1. Where this note fits

Warmup and decay schedules are the default scheduler family for modern Transformer and LLM training.

The common shape is:

```text
learning rate
^
|          peak lr
|            /\
|           /  \
|          /    \____
|_________/          \____ min lr
|
+--------------------------------> optimizer step
        warmup       decay
```

The scheduler usually advances once per `optimizer.step()`, not once per raw batch when gradient accumulation is used.

This note is organized by category so new methods can be added under the right chapter:

| Chapter            | What belongs here                                           |
| :----------------- | :---------------------------------------------------------- |
| Warmup schedulers  | Methods that increase LR at the beginning                   |
| Decay schedulers   | Methods that reduce LR after the main training phase begins |
| Composed schedules | Recipes that chain warmup and decay together                |

## 2. Warmup Schedulers

Warmup schedulers control the beginning of training. Their job is to avoid unstable early updates before the model, optimizer moments, and activations have settled.

Warmup is especially common in:

| Scenario                | Why warmup helps                               |
| :---------------------- | :--------------------------------------------- |
| LLM pretraining         | Stabilizes early large-scale training          |
| Instruction fine-tuning | Avoids damaging pretrained weights immediately |
| LoRA / QLoRA            | Lets adapters begin from a gentle update scale |
| RL fine-tuning          | Reduces early instability from noisy rewards   |

### 2.1 Linear Warmup with [`LinearLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)

Linear warmup gradually increases the learning rate from a small value to the main training learning rate.

If the optimizer's base learning rate is $\eta_{\max}$, then `LinearLR` applies a multiplicative factor to that base LR:

$$
\eta_t
= \eta_{\max}
\cdot
\left(
\text{start}_{factor}
+
\frac{t}{T_{\text{warmup}}}
(\text{end}_{factor} - \text{start}_{factor})
\right)
$$

For normal warmup:

```text
start_factor < 1
end_factor = 1
total_iters = warmup_steps
```

So the learning rate starts as:

$$
\eta_0 = \eta_{\max} \cdot \text{start}_{factor}
$$

and reaches:

$$
\eta_{\max}
$$

after the warmup window.

Minimal PyTorch usage:

```python
import torch

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

warmup_steps = 100

scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps,
)

for step in range(warmup_steps):
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

Practical example:

```text
optimizer lr = 1e-4
start_factor = 0.01
first lr = 1e-6
final warmup lr = 1e-4
```

### 2.2 Future warmup methods

Add other warmup methods here when they come up.

| Method             | Status             |
| :----------------- | :----------------- |
| Linear warmup      | Covered            |
| Constant warmup    | To add when needed |
| Exponential warmup | To add when needed |

## 3. Decay Schedulers

Decay schedulers control the middle and end of training. Their job is to reduce the update size as training moves from fast learning into refinement.

Common decay families:

| Family              | Intuition                                               |
| :------------------ | :------------------------------------------------------ |
| Constant            | Keep the same LR after warmup                           |
| Linear decay        | Reduce LR at a steady rate                              |
| Cosine decay        | Reduce LR smoothly, slow at the beginning and end       |
| Polynomial decay    | Flexible curve controlled by a power                    |
| Inverse square root | Decay slowly after warmup; classic Transformer schedule |

### 3.1 Cosine Decay with [`CosineAnnealingLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)

Cosine decay lowers the learning rate smoothly from the base learning rate to a minimum learning rate.

A common closed-form view is:

$$
\eta_t =
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max} - \eta_{\min})
\left(
1 + \cos\left(\frac{\pi t}{T_{\max}}\right)
\right)
$$

where:

| Symbol          | Meaning                                    |
| :-------------- | :----------------------------------------- |
| $\eta_t$      | Learning rate at step$t$                 |
| $\eta_{\max}$ | Starting / peak learning rate              |
| $\eta_{\min}$ | Minimum learning rate                      |
| $T_{\max}$    | Number of steps in the cosine decay window |

At the beginning:

```text
t = 0 -> lr is near eta_max
```

At the end:

```text
t = T_max -> lr is eta_min
```

Cosine decay is smooth. It does not reduce the LR too sharply at the beginning, and it becomes gentle near the end of training.

This makes it a strong default for:

| Scenario                    | Why cosine decay fits                              |
| :-------------------------- | :------------------------------------------------- |
| LLM pretraining             | Long smooth transition from learning to refinement |
| SFT                         | Stable decay over a short training run             |
| LoRA / QLoRA                | Good default when total steps are known            |
| Vision-language fine-tuning | Often works better than abrupt step drops          |

Minimal PyTorch usage:

```python
import torch

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

decay_steps = 1000
min_lr = 1e-5

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=decay_steps,
    eta_min=min_lr,
)

for step in range(decay_steps):
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

`CosineAnnealingLR` does not include warmup by itself. It starts from the optimizer's current base LR and decays toward `eta_min`.

For cosine schedules with periodic restarts, use `CosineAnnealingWarmRestarts`, which belongs in [Cyclical and Restart](./Cyclical_and_Restart.md).

### 3.2 Future decay methods

Add other decay methods here when they come up.

| Method                    | Status             |
| :------------------------ | :----------------- |
| Cosine decay              | Covered            |
| Constant LR after warmup  | To add when needed |
| Linear decay              | To add when needed |
| Polynomial decay          | To add when needed |
| Inverse-square-root decay | To add when needed |

## 4. Composed Warmup + Decay Schedules

Many practical schedules are not a single scheduler. They are a warmup scheduler
followed by a decay scheduler.

In LLM training, "cosine schedule" often means:

```text
linear warmup first, cosine decay after warmup
```

PyTorch represents this as:

1. `LinearLR` for warmup.
2. `CosineAnnealingLR` for decay.
3. `SequentialLR` to switch from warmup to decay.

### 4.1 Linear Warmup + Cosine Decay with [`SequentialLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html)

Minimal PyTorch usage:

```python
import torch

model = torch.nn.Linear(10, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

total_steps = 1000
warmup_steps = 100
decay_steps = total_steps - warmup_steps
min_lr = 1e-5

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps,
)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=decay_steps,
    eta_min=min_lr,
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[warmup_steps],
)

for step in range(total_steps):
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## 5. How to categorize new schedulers

Use the scheduler's main behavior:

| Behavior                               | Put it in                                        |
| :------------------------------------- | :----------------------------------------------- |
| LR increases at the beginning          | Warmup schedulers                                |
| LR decreases over training             | Decay schedulers                                 |
| Warmup is chained with decay           | Composed warmup + decay schedules                |
| LR repeatedly rises again              | [Cyclical and Restart](./Cyclical_and_Restart.md) |
| LR changes based on validation metrics | [Metric Adaptive](./Metric_Adaptive.md)           |

## 6. Key takeaways

`LinearLR` is the natural PyTorch scheduler for linear warmup when configured
with `start_factor < 1` and `end_factor = 1`.

`CosineAnnealingLR` is the natural PyTorch scheduler for cosine decay when
configured with `T_max = decay_steps` and an optional `eta_min`.

For warmup plus cosine decay, combine them with `SequentialLR`.
