# Rotary Position Embeddings (RoPE)

## Principle: Why RoPE Dominates Modern LLMs
Absolute positional encodings (original Transformer sinusoidal or learned) add a fixed vector per position → poor extrapolation beyond training length.

Relative encodings (e.g., T5 bias, ALiBi) add biases based on distance → better extrapolation but can distort attention patterns.

**RoPE** elegantly encodes **relative positions** by rotating query/key vectors in 2D planes. The inner product becomes a function of $|m-n|$ only → perfect relative bias, length extrapolation and no extra parameters.

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
The attention score $q_m^T R((n-m)\Theta) k_n$ **depends only on the relative position** $m-n$, not on absolute positions $m$ or $n$.

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
```python
class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE encodes positional information by rotating query and key vectors
    in 2D planes, making attention scores depend only on relative positions.
    
    Original Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Attributes:
        dim (int): Dimension of the input features
        base (int): Base for frequency calculation (default: 10000)
        max_seq_len (int): Maximum sequence length for precomputing frequencies
    """
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 512):
        """
        Initialize Rotary Position Embedding.
        
        Args:
            dim: Dimension of input features (must be even)
            base: Base for frequency calculation
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        assert dim % 2 == 0, f"Dimension must be even, got {dim}"

        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Precompute frequencies and rotation angles
        self._precompute_frequencies()
        
    def _precompute_frequencies(self):
        """Precompute frequencies for all positions and dimensions."""
        # Calculate frequencies for each dimension pair
        # θ_j = base^(-2j/d) for j = 0, 1, ..., d/2-1
        j = torch.arange(0, self.dim, 2, dtype=torch.float32)
        theta = 1.0 / (self.base ** (j / self.dim))

        # Precompute sin and cos for all positions
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32)

        # Create position-frequency matrix: pos * theta
        # Shape: (max_seq_len, dim/2)
        m_theta = positions.unsqueeze(1) * theta.unsqueeze(0)

        # Precompute cos and sin values
        # Shape: (max_seq_len, dim)
        cos_cached = torch.cos(m_theta).repeat_interleave(2, dim=1)
        sin_cached = torch.sin(m_theta).repeat_interleave(2, dim=1)

        # Register as buffers (not trainable parameters)
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions for RoPE implementation.
        
        For a tensor shaped (..., d), this function rearranges it as:
        from [x_{2i}, x_{2i+1}] to [-x_{2i+1}, x_{2i}]
        to implement complex rotation.
        
        Args:
            x: Input tensor of shape (..., d)
            
        Returns:
            Rotated tensor of same shape
        """
        d = x.shape[-1]
        x_reshaped = x.view(*x.shape[:-1], d//2, 2)
        x1 = x_reshaped[..., 0]     # x_{2i}
        x2 = x_reshaped[..., 1]     # x_{2i+1}
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.view(*x.shape)
    
    def apply_rotary_pos_emb(
        self, 
        x: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        The transformation is: x' = x * cos(pos*theta) + rotate_half(x) * sin(pos*theta)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            positions: Position indices for each token in sequence.
                      If None, use sequential positions [0, 1, ..., seq_len-1]
                      
        Returns:
            Tensor with rotary position encoding applied
        """
        batch_size, seq_len, num_heads, head_dim = x.shape

        # Get position indices
        if positions is None:
            positions = torch.arange(0, seq_len, device=x.device)
        else:
            # Ensure positions are within bounds
            positions = positions.clamp(0, self.max_seq_len-1)

        # Reshape for broadcasting: (1, seq_len, 1, dim)
        cos = self.cos_cached[positions].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
        sin = self.sin_cached[positions].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
        
        # Expand to match input tensor shape (batch_size, seq_len, num_heads, dim)
        cos = cos.expand(batch_size, -1, num_heads, -1)
        sin = sin.expand(batch_size, -1, num_heads, -1)

        # Apply RoPE formula: x_rotated = x * cos + rotate_half(x) * sin
        x_rotated = x * cos + self._rotate_half(x) * sin
        
        return x_rotated
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim)
            positions: Position indices for each token
            
        Returns:
            Tuple of (q_rotated, k_rotated) with same shapes as input
        """
        q_rotated = self.apply_rotary_pos_emb(q, positions)
        k_rotated = self.apply_rotary_pos_emb(k, positions)
        
        return q_rotated, k_rotated
```