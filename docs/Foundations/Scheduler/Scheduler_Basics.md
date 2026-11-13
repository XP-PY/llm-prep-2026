# Learning Rate Scheduler Basics

## 1. Why this note exists

The optimizer decides how to update parameters from gradients. The learning rate scheduler decides how large the global update scale should be at each training step.

For AdamW, a simplified update is:

$$
\theta_t
= \theta_{t-1}
- \eta_t \cdot \text{AdamStep}(g_t)
- \eta_t \lambda \theta_{t-1}
$$

Here, $\eta_t$ is the learning rate at step $t$. A scheduler is simply a rule for choosing $\eta_t$ over time.

The important separation is:

| Component         | Main job                                           |
| :---------------- | :------------------------------------------------- |
| Optimizer         | Computes the update direction and adaptive scaling |
| Scheduler         | Changes the global step size during training       |
| Weight decay      | Regularizes parameter magnitude                    |
| Gradient clipping | Limits unusually large updates                     |

A bad schedule can make a good optimizer unstable. A good schedule can make the same optimizer train smoothly, especially for large Transformer models.

## 2. The core problem

Training does not need the same learning rate at every stage.

**Early training** is fragile because weights are random or only weakly adapted to the new task. A large learning rate can create unstable activations, loss spikes, or divergence.

**Middle training** usually benefits from a larger learning rate because the model has found a useful region and can make faster progress.

**Late training** often benefits from a smaller learning rate because the model is refining behavior. Large updates near the end can overwrite useful structure.

This is why many schedules have this shape:

```text
learning rate
^
|          _________
|         /         \
|        /           \
|_______/             \____
|
+--------------------------------> training step
        warmup        decay
```

## 3. Step count is the scheduler clock

Most modern LLM training schedules are defined over optimizer steps, not raw mini-batches.

If gradient accumulation is used:

```text
optimizer_steps = raw_batches / gradient_accumulation_steps
```

Example:

```text
raw batches = 10000
gradient accumulation steps = 8
optimizer steps = 1250
```

The scheduler should usually step when `optimizer.step()` runs, not every time `loss.backward()` runs. Otherwise the learning rate changes too quickly.

## 4. Basic scheduler vocabulary

| Term             | Meaning                                                     |
| :--------------- | :---------------------------------------------------------- |
| Base LR / max LR | Main peak learning rate used after warmup                   |
| Initial LR       | Learning rate at the first optimizer step                   |
| Warmup steps     | Steps spent increasing LR from a small value to the base LR |
| Decay steps      | Steps spent reducing LR after warmup                        |
| Total steps      | Planned number of optimizer updates                         |
| Min LR           | Lowest LR allowed near the end of training                  |
| Warmup ratio     | `warmup_steps / total_steps`                              |
| Schedule shape   | Linear, cosine, inverse-square-root, polynomial, etc.       |

## 5. Warmup

Warmup gradually increases the learning rate at the beginning of training.

A common linear warmup is:

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}
$$

where:

| Symbol                | Meaning                              |
| :-------------------- | :----------------------------------- |
| $t$                 | Current optimizer step during warmup |
| $\eta_{\max}$       | Peak learning rate                   |
| $T_{\text{warmup}}$ | Number of warmup steps               |

Warmup is especially important for Transformers because attention, residual connections, normalization, and adaptive optimizers can produce unstable early updates.

Common warmup choices:

| Setting                  | Typical warmup                               |
| :----------------------- | :------------------------------------------- |
| Large-scale pretraining  | Thousands of steps                           |
| Instruction fine-tuning  | 1% to 5% of total steps                      |
| LoRA / QLoRA fine-tuning | Often short, such as 3% to 10%               |
| Tiny experiments         | Sometimes no warmup, but still worth testing |

## 6. Decay

Decay reduces the learning rate after the main training phase begins.

The purpose is not only to avoid divergence. Decay also changes the training behavior from exploration to refinement.

Common decay families:

| Family              | Intuition                                               |
| :------------------ | :------------------------------------------------------ |
| Constant            | Keep the same LR after warmup                           |
| Linear decay        | Reduce LR at a steady rate                              |
| Cosine decay        | Reduce LR smoothly, slow at the beginning and end       |
| Polynomial decay    | Flexible curve controlled by a power                    |
| Inverse square root | Decay slowly after warmup; classic Transformer schedule |

These belong in [Warmup and Decay](./Warmup_and_Decay.md).

## 7. Schedule families in this repo

This folder groups schedulers by behavior instead of using one file per scheduler.

| Note                                             | What it should cover                                                           |
| :----------------------------------------------- | :----------------------------------------------------------------------------- |
| [Warmup and Decay](./Warmup_and_Decay.md)         | Constant, linear, cosine, polynomial, inverse-square-root, warmup combinations |
| [Cyclical and Restart](./Cyclical_and_Restart.md) | Cyclical LR, cosine restart, OneCycle-style schedules                          |
| [Metric Adaptive](./Metric_Adaptive.md)           | Reduce-on-plateau, patience, validation-driven LR changes                      |
| [LLM Training Recipes](./LLM_Training_Recipes.md) | Practical schedules for pretraining, SFT, LoRA, QLoRA, and RL fine-tuning      |

## 8. Step-level vs epoch-level scheduling

There are two common ways to update a scheduler:

| Mode        | Meaning                        | Common use                         |
| :---------- | :----------------------------- | :--------------------------------- |
| Step-level  | Update LR every optimizer step | LLM pretraining and fine-tuning    |
| Epoch-level | Update LR once per epoch       | Smaller supervised learning setups |

For LLM work, step-level scheduling is usually the default because datasets are large, epochs may be poorly defined, and training is planned by token budget or optimizer steps.

## 9. Interaction with batch size

Learning rate is tied to effective batch size.

```text
effective_batch_size = per_device_batch_size
                     * number_of_devices
                     * gradient_accumulation_steps
```

When the effective batch size changes, the best learning rate may also change. A common rough rule is that larger batches can often tolerate larger learning rates, but this is only a starting point. Warmup length, optimizer betas, model size, data quality, and task difficulty also matter.

## 10. Interaction with AdamW

Schedulers are commonly paired with AdamW because AdamW handles adaptive per-parameter scaling while the scheduler handles global training phases.

Important details:

1. AdamW's `lr` is still very important even though AdamW is adaptive.
2. Decoupled weight decay is usually multiplied by the current LR.
3. LR warmup often stabilizes AdamW's early moment estimates.
4. A too-large peak LR can still destroy training.
5. A too-small LR can make training look stable but underfit.

See [AdamW](../Optimizer/AdamW.md) for the optimizer side.

## 11. Minimal PyTorch mental model

The usual order is:

```python
loss.backward()
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

With gradient accumulation, only step the optimizer and scheduler after the accumulation window:

```python
loss = loss / gradient_accumulation_steps
loss.backward()

if (batch_idx + 1) % gradient_accumulation_steps == 0:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

The exact order can vary for some PyTorch schedulers, but the main principle is that the scheduler clock should match optimizer updates.

## 12. Common mistakes

### Stepping the scheduler too often

If `scheduler.step()` runs every raw batch while `optimizer.step()` runs only after gradient accumulation, the LR schedule finishes too early.

### Computing total steps incorrectly

Total steps should usually mean optimizer steps:

```text
total_steps = epochs * batches_per_epoch / gradient_accumulation_steps
```

For distributed training, `batches_per_epoch` should already reflect the per-rank dataloader length.

### Forgetting warmup for large models

Large Transformer training often becomes unstable without warmup, especially when the peak LR is aggressive.

### Using validation-driven schedules for huge pretraining

Metric-based schedules are useful for many smaller experiments, but they are less natural for large LLM pretraining, where validation is expensive and training is usually planned by tokens or steps.

### Comparing runs with different schedule lengths

Two runs with the same peak LR but different total steps can have very different average learning rates.

## 13. Practical starting points

| Scenario                             | Reasonable first schedule             |
| :----------------------------------- | :------------------------------------ |
| LLM pretraining                      | Linear warmup + cosine decay          |
| Instruction SFT                      | Short warmup + cosine or linear decay |
| LoRA / QLoRA                         | Short warmup + cosine or constant LR  |
| Small supervised task                | Cosine decay or reduce-on-plateau     |
| Reproducing Transformer paper basics | Warmup + inverse-square-root decay    |

These are starting points, not universal rules.

## 14. What to learn next

Recommended order:

1. Learn warmup, constant LR, linear decay, and cosine decay.
2. Understand inverse-square-root because it appears in classic Transformer training.
3. Learn metric-adaptive schedules for smaller experiments.
4. Learn cyclical schedules and restarts as optional tools.
5. Build practical recipes for LLM pretraining, SFT, LoRA, QLoRA, and RL.

## 15. Key takeaways

The shortest mental model:

```text
optimizer = how to use gradients
scheduler = how hard to step over time
warmup = avoid unstable early updates
decay = move from fast learning to refinement
```

For LLMs, the most important scheduler pattern is:

```text
linear warmup + long decay over optimizer steps
```
