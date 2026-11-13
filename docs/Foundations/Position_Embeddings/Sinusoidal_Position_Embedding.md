# [Sinusoidal Position Embedding](https://arxiv.org/abs/1706.03762)

## Principle: Why Transformers Need Position Embeddings

Self-attention sees a sequence as a set of token vectors. Without position
information, the attention operation itself does not know whether a token is at
the beginning, middle, or end of the sequence.

For example, these two sequences contain the same tokens but mean different
things:

```text
dog bites man
man bites dog
```

A Transformer therefore needs a way to inject token order.

Sinusoidal position embedding is the fixed, parameter-free method used in the
original Transformer. It creates one deterministic vector for each absolute
position and adds that vector to the token embedding.

$$
x_{\text{input}}(p) = x_{\text{token}}(p) + PE(p)
$$

where:

| Symbol | Meaning |
|:---|:---|
| $p$ | Token position in the sequence |
| $x_{\text{token}}(p)$ | Token embedding at position $p$ |
| $PE(p)$ | Sinusoidal position embedding at position $p$ |

## 1. Core Formula

For model dimension $d_{\text{model}}$, sinusoidal position embedding defines:

$$
PE(p, 2i) = \sin\left(\frac{p}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE(p, 2i + 1) = \cos\left(\frac{p}{10000^{2i / d_{\text{model}}}}\right)
$$

where:

| Symbol | Meaning |
|:---|:---|
| $p$ | Sequence position: $0, 1, 2, \dots, L-1$ |
| $i$ | Dimension-pair index: $0, 1, \dots, d_{\text{model}}/2 - 1$ |
| $2i$ | Even embedding dimension |
| $2i + 1$ | Odd embedding dimension |
| $10000$ | Base used to create many frequencies |

The even dimensions use sine. The odd dimensions use cosine.

## 2. Dimension Pairs

The embedding dimensions are processed in pairs:

```text
(0, 1), (2, 3), (4, 5), ..., (d_model - 2, d_model - 1)
```

For each pair $i$:

```text
dimension 2i     -> sine channel
dimension 2i + 1 -> cosine channel
```

So a position vector looks like:

$$
PE(p) =
\left[
\sin(p\theta_0),
\cos(p\theta_0),
\sin(p\theta_1),
\cos(p\theta_1),
\dots,
\sin(p\theta_{d/2-1}),
\cos(p\theta_{d/2-1})
\right]
$$

where:

$$
\theta_i = 10000^{-2i / d_{\text{model}}}
$$

Equivalently:

$$
p\theta_i = \frac{p}{10000^{2i / d_{\text{model}}}}
$$

## 3. Calculation Process

Suppose:

```text
sequence length L = 4
d_model = 6
```

The position indices are:

$$
p = [0, 1, 2, 3]
$$

The dimension-pair indices are:

$$
i = [0, 1, 2]
$$

### Step 1: Compute frequency for each dimension pair

For each pair $i$, compute:

$$
\theta_i = 10000^{-2i / d_{\text{model}}}
$$

With $d_{\text{model}} = 6$:

$$
\theta_0 = 10000^{-0/6} = 1
$$

$$
\theta_1 = 10000^{-2/6}
$$

$$
\theta_2 = 10000^{-4/6}
$$

So lower dimensions change quickly, while higher dimensions change slowly.

### Step 2: Compute the angle matrix

For every position $p$ and dimension pair $i$:

$$
A[p, i] = p \cdot \theta_i
$$

The shape is:

```text
A.shape = [sequence_length, d_model / 2]
```

For $L = 4$ and $d_{\text{model}} = 6$:

```text
A.shape = [4, 3]
```

Conceptually:

$$
A =
\begin{bmatrix}
0\theta_0 & 0\theta_1 & 0\theta_2 \\
1\theta_0 & 1\theta_1 & 1\theta_2 \\
2\theta_0 & 2\theta_1 & 2\theta_2 \\
3\theta_0 & 3\theta_1 & 3\theta_2
\end{bmatrix}
$$

### Step 3: Apply sine to even dimensions

Fill dimensions `0, 2, 4, ...` with:

$$
\sin(A[p, i])
$$

So:

```text
PE[:, 0] = sin(A[:, 0])
PE[:, 2] = sin(A[:, 1])
PE[:, 4] = sin(A[:, 2])
```

### Step 4: Apply cosine to odd dimensions

Fill dimensions `1, 3, 5, ...` with:

$$
\cos(A[p, i])
$$

So:

```text
PE[:, 1] = cos(A[:, 0])
PE[:, 3] = cos(A[:, 1])
PE[:, 5] = cos(A[:, 2])
```

### Step 5: Add position embeddings to token embeddings

If token embeddings have shape:

```text
token_embeddings.shape = [batch_size, seq_len, d_model]
```

and the position table has shape:

```text
position_embeddings.shape = [seq_len, d_model]
```

then broadcasting gives:

$$
X = X_{\text{token}} + PE
$$

Shape:

```text
X.shape = [batch_size, seq_len, d_model]
```

## 4. Why Use Many Frequencies?

Each dimension pair uses a different wavelength.

Low dimension pairs use high-frequency waves:

```text
position changes quickly -> embedding value changes quickly
```

High dimension pairs use low-frequency waves:

```text
position changes slowly -> embedding value changes slowly
```

This gives the model both fine-grained and long-range position signals.

| Dimension region | Frequency | What it captures |
|:---|:---|:---|
| Lower dimensions | Higher frequency | Nearby position differences |
| Higher dimensions | Lower frequency | Broader sequence position trends |

## 5. Relative Position Intuition

Sinusoidal embeddings are absolute position embeddings because each position
gets a fixed vector.

However, they also have a useful relative-position property. Because of sine and
cosine angle identities, the embedding at position $p+k$ can be represented as a
linear function of the embedding at position $p$.

The key identities are:

$$
\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)
$$

$$
\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)
$$

For a fixed offset $k$:

$$
\sin((p+k)\theta_i)
=
\sin(p\theta_i)\cos(k\theta_i)
+
\cos(p\theta_i)\sin(k\theta_i)
$$

$$
\cos((p+k)\theta_i)
=
\cos(p\theta_i)\cos(k\theta_i)
-
\sin(p\theta_i)\sin(k\theta_i)
$$

This means relative offsets can be inferred through linear transformations of
the sine/cosine pair.

Important distinction:

| Method | How position is applied |
|:---|:---|
| Sinusoidal position embedding | Add fixed absolute position vector to token embedding |
| RoPE | Rotate query and key vectors so attention scores directly depend on relative distance |

See [RoPE](./RoPE.md) for the rotary version.

## 6. Benefits and Limitations

| Aspect | Sinusoidal position embedding |
|:---|:---|
| Extra parameters | None |
| Position type | Absolute |
| Added to token embeddings | Yes |
| Can generate positions beyond training length | Yes, because formula is deterministic |
| Strong long-context extrapolation | Limited compared with modern methods |
| Common in current LLMs | Mostly replaced by RoPE or relative methods |

The main limitation is that position is injected only once at the input. Modern
LLMs usually prefer RoPE because relative position information directly affects
attention scores at every layer.

## 7. Step-by-Step Python Implementation

### Minimal function

```python
import torch


def sinusoidal_position_embedding(
    seq_len: int,
    d_model: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build sinusoidal position embeddings.

    Args:
        seq_len: Sequence length.
        d_model: Embedding dimension. Must be even.
        base: Frequency base used in the original Transformer.
        device: Optional torch device.

    Returns:
        Tensor with shape [seq_len, d_model].
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}.")

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    pair_indices = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)

    # inv_freq[i] = base^(-2i / d_model), where pair_indices = 2i.
    inv_freq = base ** (-pair_indices / d_model)

    # angles[p, i] = p * inv_freq[i]
    angles = positions[:, None] * inv_freq[None, :]

    embeddings = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)
    embeddings[:, 0::2] = torch.sin(angles)
    embeddings[:, 1::2] = torch.cos(angles)
    return embeddings
```

### Module version

```python
import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Fixed sinusoidal position embedding.

    This module precomputes a [max_seq_len, d_model] table and adds the matching
    position vectors to input token embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}.")

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.dropout = nn.Dropout(dropout)

        position_embedding = self._build_table(max_seq_len, d_model, base)

        # Register as a buffer, not a parameter. It moves with .to(device), but
        # it is not optimized by gradient descent.
        self.register_buffer("position_embedding", position_embedding, persistent=False)

    @staticmethod
    def _build_table(seq_len: int, d_model: int, base: float) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.float32)
        pair_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
        inv_freq = base ** (-pair_indices / d_model)
        angles = positions[:, None] * inv_freq[None, :]

        table = torch.zeros(seq_len, d_model, dtype=torch.float32)
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)
        return table

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: Tensor with shape [batch, seq_len, d_model].

        Returns:
            Tensor with shape [batch, seq_len, d_model].
        """
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape [batch, seq_len, d_model], "
                f"got {tuple(token_embeddings.shape)}."
            )

        _, seq_len, d_model = token_embeddings.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        positions = self.position_embedding[:seq_len].unsqueeze(0)
        return self.dropout(token_embeddings + positions)
```

### Usage example

```python
batch_size = 2
seq_len = 8
d_model = 16

token_embeddings = torch.randn(batch_size, seq_len, d_model)
position_layer = SinusoidalPositionEmbedding(d_model=d_model, max_seq_len=128)

inputs = position_layer(token_embeddings)

print(inputs.shape)
# torch.Size([2, 8, 16])
```

## 8. Key Takeaways

Sinusoidal position embedding builds a fixed table:

```text
[seq_len, d_model]
```

Each position gets sine/cosine values across many frequencies.

The calculation flow is:

```text
positions -> frequencies -> angle matrix -> sin/cos channels -> add to tokens
```

It is simple, parameter-free, and historically important, but modern LLMs
usually prefer RoPE because it injects relative position structure directly into
attention.
