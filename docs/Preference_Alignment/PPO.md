# Proximal Policy Optimization (PPO) for RLHF

## 🎯 What is PPO?
**Proximal Policy Optimization** - A stable RL algorithm that prevents destructive large policy updates.

## 📊 The Core Idea
**"Take small, safe steps"** - Clip policy changes to avoid breaking the model.

## 🔧 Key Components in RLHF

### 4 Models You Need:
```python
1. Policy Model (π_θ)      # LLM we're optimizing
2. Reference Model (π_ref) # Frozen SFT model
3. Reward Model (r_φ)      # Gives scores
4. Value Model (V_ψ)       # Estimates state value
```

## 🎯 Reward Function
```
Total Reward = RM Score - β × KL Penalty
```
Where:
- **RM Score**: Reward model prediction
- **KL Penalty**: KL(π_θ || π_ref) - prevents drifting too far
- **β**: Tuning parameter (usually 0.1-0.2)

## ⚡ PPO's Magic: Clipping

### The Clipped Objective:
```python
ratio = π_new(action|state) / π_old(action|state)

# Clipped loss:
L_clip = min(ratio × Advantage, 
             clip(ratio, 1-ε, 1+ε) × Advantage)
```
- **ε** (epsilon): Clip range (typically 0.2)
- **Advantage**: How much better than average

## 🔄 Training Steps

### 1. Generate Data
```python
response = policy_model.generate(prompt)
logprobs = get_log_probabilities(response)
```

### 2. Compute Rewards
```python
rm_score = reward_model(prompt, response)
kl_penalty = KL(policy_logits, ref_logits)
reward = rm_score - beta * kl_penalty
```

### 3. Calculate Advantages
```python
# Using Generalized Advantage Estimation (GAE)
advantages = compute_gae(rewards, values)
```

### 4. PPO Update
```python
# Compute probability ratio
ratio = exp(new_logprob - old_logprob)

# Clipped loss
loss = -min(ratio * advantage, 
           clip(ratio, 0.8, 1.2) * advantage)
```

## 📈 Hyperparameters

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| ε (epsilon) | 0.2 | Clipping range |
| β (beta) | 0.1 | KL penalty weight |
| γ (gamma) | 0.99 | Discount factor |
| λ (lambda) | 0.95 | GAE parameter |
| Learning Rate | 1e-6 | Very small for stability |

## 💡 Why PPO Works for RLHF?

### 1. **Stability**
- Clipping prevents catastrophic updates
- No more "one bad update ruins everything"

### 2. **Sample Efficiency**
- Reuse data multiple times (PPO epochs > 1)
- Better than vanilla policy gradient

### 3. **Automatic KL Control**
- Reference model keeps outputs natural
- No need for complex KL constraints

## ⚠️ Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| Reward hacking | Increase β, normalize rewards |
| Training instability | Gradient clipping, LR scheduling |
| High memory usage | Gradient checkpointing, mixed precision |

## 🚀 Simple Code Skeleton

```python
# Simplified PPO for RLHF
class PPOTrainer:
    def train_step(self, prompts):
        # 1. Generate responses
        responses = self.policy.generate(prompts)
        
        # 2. Get rewards
        rm_scores = self.reward_model(prompts, responses)
        kl = KL(self.policy, self.ref_model)
        rewards = rm_scores - self.beta * kl
        
        # 3. PPO update
        advantages = compute_advantages(rewards)
        loss = ppo_clip_loss(advantages, epsilon=0.2)
        
        # 4. Optimize
        loss.backward()
        optimizer.step()
```

## 📊 Monitoring Metrics

Track these during training:
1. **Policy Loss** - Should decrease smoothly
2. **KL Divergence** - Should stay small (< 10)
3. **Reward** - Should increase gradually
4. **Clip Fraction** - % of samples clipped (ideally 10-30%)

## 🎯 Quick Tips

1. **Start small**: β = 0.1, ε = 0.2, LR = 1e-6
2. **Monitor KL**: Keep it between 1-20 nats
3. **Watch clipping**: If >50% samples clipped, reduce LR
4. **Use warmup**: Gradually increase batch size

## 📚 TL;DR

**PPO = Policy Gradient + Clipping + KL Penalty**

It's the **go-to algorithm for RLHF** because:
- ✅ Stable training
- ✅ Preserves text quality
- ✅ Prevents reward hacking
- ✅ Works with LLMs

---

*Remember: In RLHF, we're not just maximizing reward - we're finding the sweet spot between getting high scores and staying human-like!* 🤖→👤