# Rotary Position Embeddings (RoPE)

## Principle: Why RoPE Dominates Modern LLMs
Absolute positional encodings (original Transformer sinusoidal or learned) add a fixed vector per position → <span style='color : blue'>poor extrapolation beyond training length</span>.

Relative encodings (e.g., T5 bias, ALiBi) add biases based on distance → <span style='color : blue'>better extrapolation but can distort attention patterns</span>.

**RoPE** elegantly encodes <span style='color : blue'>**relative positions**</span> by rotating query/key vectors in 2D planes. The inner product becomes a function of $|m-n|$ only 
→ <span style='color : blue'>perfect relative bias, length extrapolation and no extra parameters</span>.

Real-world impact:
* LLaMA-3 uses RoPE for 128k context.
* PaLM/Gemma extend it (e.g., RoPE + NTK scaling for even longer).
* Pitfall of alternatives: Absolute sinusoidal fails hard on extrapolation; ALiBi can reduce effective rank.

## Detailed Mathematical Formulation
### **Step 1: Vector Grouping**
For a $d$-dimensional vector $x_m$, we group every two dimensions into a complex number (2D plane):

$$
x_m = [x_0, x_1, x_2, x_3, \dots, x_{d-2}, x_{d-1}]
$$

**Grouped as:** $(x_0, x_1), (x_2, x_3), \dots, (x_{d-2}, x_{d-1})$

---

### **Step 2: Position-dependent Rotation Angles**
For position $m$, the rotation angle for the $j$-th group (dimension pair $(2j, 2j+1)$) is:

$$
\theta_j = 10000^{-2j/d} \quad \text{(frequency decreases with dimension)}
$$

The rotation angle for position $m$ at group $j$ is:

$$
m\theta_j = m \cdot 10000^{-2j/d}
$$

---

### **Step 3: Rotation Matrix**
In 2D plane, the rotation matrix with angle $m\theta_j$ is:

$$
R(m\theta_j) = \begin{bmatrix}
\cos(m\theta_j) & -\sin(m\theta_j) \\
\sin(m\theta_j) & \cos(m\theta_j)
\end{bmatrix}
$$

This matrix acts on the $j$-th dimension pair:

$$
\begin{bmatrix}
x_{2j}^{\text{rope}} \\
x_{2j+1}^{\text{rope}}
\end{bmatrix}
= R(m\theta_j)
\begin{bmatrix}
x_{2j} \\
x_{2j+1}
\end{bmatrix}
$$

---

### **Step 4: Complete RoPE Transformation**
For the entire vector, RoPE transformation is a block-diagonal matrix:

$$
x_m^{\text{rope}} = 
\begin{bmatrix}
R(m\theta_0) & 0 & \cdots & 0 \\
0 & R(m\theta_1) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & R(m\theta_{d/2-1})
\end{bmatrix}
x_m
$$

We denote this as: $x_m^{\text{rope}} = R(m\Theta) x_m$, where $R(m\Theta)$ represents the entire block-diagonal matrix.

---

### **Step 5: Key Derivation - Attention Scores**
Now we examine how the dot product between Query and Key depends only on relative position.

Let **Query** be at position $m$ and **Key** at position $n$:

$$
q_m^{\text{rope}} = R(m\Theta) q_m
$$
$$
k_n^{\text{rope}} = R(n\Theta) k_n
$$

**Compute the dot product:**

$$
\begin{aligned}
(q_m^{\text{rope}})^T k_n^{\text{rope}} &= (R(m\Theta) q_m)^T (R(n\Theta) k_n) \\
&= q_m^T R(m\Theta)^T R(n\Theta) k_n
\end{aligned}
$$

---

### **Step 6: Important Properties of Rotation Matrices**
1. **Orthogonality:** $R(\theta)^T R(\theta) = I$
2. **Transpose = Negative rotation:** $R(\theta)^T = R(-\theta)$
3. **Composition:** $R(\alpha) R(\beta) = R(\alpha + \beta)$

Therefore:

$$
R(m\Theta)^T R(n\Theta) = R(-m\Theta) R(n\Theta) = R((n-m)\Theta)
$$

---

### **Step 7: Final Result**
Substituting back:

$$
(q_m^{\text{rope}})^T k_n^{\text{rope}} = q_m^T R((n-m)\Theta) k_n
$$

Equivalently (using $R(-\theta) = R(\theta)^T$):

$$
= q_m^T R((m-n)\Theta)^T k_n
$$

---

### **Step 8: Why This is Perfect Relative Position Encoding?**

#### **Mathematically**
The attention score $q_m^T R((n-m)\Theta) k_n$ <span style='color : blue'>**depends only on the relative position**</span> $m-n$, not on absolute positions $m$ or $n$.

#### **Physical Interpretation**
If we view vectors as rotations in 2D planes:

- **Position $m$:** rotated by $m\theta_j$
- **Position $n$:** rotated by $n\theta_j$
- **Their relative angle:** $n\theta_j - m\theta_j = (n-m)\theta_j$

This relative angle encodes the relative position information!

### **Summary**

**RoPE** encodes positions through rotation, making attention scores depend solely on relative positions $|m-n|$, achieving perfect relative position bias in the attention mechanism.

**One-sentence intuition:** RoPE treats position encoding as rotating the vector in multiple 2D planes, where the rotation difference between positions captures their relative distance.

## Comparison Table
| Encoding Type | Absolute/Relative | Extrapolation Performance | Parameters | Bias Form | Length Limit | Used In |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Sinusoidal (absolute) | Absolute|Poor (fails > train len)|0|Fixed vector add|~Train length|Original Transformer|
|Learned absolute|Absolute|Poor|d|Fixed vector add|~Train length|Early BERT/GPT|
|ALiBi |Relative (bias)|Good|0|-slope *|i-j|Some long-context models|
|RoPE|Relative (via rotation)|Excellent (theoretical guarantee)|0|Rotation matrix on Q/K|32k–128k (with NTK)|LLaMA, PaLM, Grok, Gemma|
|xPos |Relative (advanced rotary)|Superior|0|RoPE + shrinkage γ^|i-j||

## Appendix

### Neural Tangent Kernel (NTK)
#### Questions from RoPE
RoPE works well within the training length, but when encountering sequences that far exceed the training length:
1. **High-frequency dimensional collapse:** The rotation frequency in high dimensions is too fast, causing positional differences to become indistinct.
2. **Degraded extrapolation performance:** The model fails to generalize to locations beyond the training length.
#### The core idea of ​​NTK-aware scaling
Adjusting the frequency distribution of position encoding by **scaling the base frequency**:
```python
# Original RoPE frequency calculation (Llama)
base = 10000.0
theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

# NTK-aware scaling
def ntk_scaled_rope(dim, seq_len, base=10000.0, scaling_factor=1.0):
    """
    dim: hidden dim
    seq_len: present seq length
    base: original base frequency
    scaling_factor: scaling factor
    """
    # Caculate base frequency after Scaling
    scaled_base = base * scaling_factor ** (dim / (dim - 2))
    theta = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
    return theta
```

### [ALiBi](https://arxiv.org/abs/2108.12409)
Add linear bias to attention scores (no embeddings):
$$S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}} - m \cdot |i - j|$$
m: Head-specific slope (e.g., 2^{-8/head_idx}).

→ Negative bias for distant tokens → soft window.

### [xPos](https://arxiv.org/abs/2212.10554)
Builds on RoPE with positional shrinkage:
$$\text{xPos}(m) = \exp(\gamma (m - L/2)) \cdot \text{RoPE}(m)$$
γ < 0: Decay for positions > L/2.

Relative form preserves RoPE's rotation while shrinking distant contributions → stable extrapolation to 10x+ lengths.

## Step-by-Step Code Implementation
[Python script](../../src/part1_positional_encodings.ipynb)