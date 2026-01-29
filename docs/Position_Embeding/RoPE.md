# Rotary Position Embeddings (RoPE)

## Principle: Why RoPE Dominates Modern LLMs
Absolute positional encodings (original Transformer sinusoidal or learned) add a fixed vector per position → <span style='color : aqua'>poor extrapolation beyond training length</span>.

Relative encodings (e.g., T5 bias, ALiBi) add biases based on distance → <span style='color : aqua'>better extrapolation but can distort attention patterns</span>.

**RoPE** elegantly encodes <span style='color : aqua'>**relative positions**</span> by rotating query/key vectors in 2D planes. The inner product becomes a function of $|m-n|$ only 
→ <span style='color : aqua'>perfect relative bias, length extrapolation and no extra parameters</span>.

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
The attention score $q_m^T R((n-m)\Theta) k_n$ <span style='color : aqua'>**depends only on the relative position**</span> $m-n$, not on absolute positions $m$ or $n$.

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
| Encoding Type | Absolute/Relative | Extrapolation Performance | Parameters | Used In |
|:---:|:---:|:---:|:---:|:---:|
| Sinusoidal (absolute) | Absolute|Poor (fails > train len)|0|Original Transformer|
Learned absolute|Absolute|Poor|d|Early BERT/GPT|
ALiBi (bias)|Relative|Good|0|Some long-context models|
RoPE|Relative (via rotation)|Excellent (theoretical guarantee)|0|"LLaMA, PaLM, Grok, Gemma"|

## Step-by-Step Code Implementation
[Python script](../../src/week1_positional_encodings.ipynb)