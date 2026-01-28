# [SVD (Singular Value Decomposition)](https://en.wikipedia.org/wiki/Singular_value_decomposition)

## 1. Formal Definition

For any real matrix $A \in \mathbb{R}^{m \times n}$ (not necessarily square), the **Singular Value Decomposition** factorizes it as:
$$
A = U \Sigma V^T
$$
where:
- $U \in \mathbb{R}^{m \times m}$: orthogonal matrix ($U^T U = I$), columns are **left singular vectors**.
- $ \Sigma \in \mathbb{R}^{m \times n} $: rectangular diagonal matrix with non-negative singular values $ \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 $ on the diagonal (r = rank(A)), zeros elsewhere.
- $ V \in \mathbb{R}^{n \times n} $: orthogonal matrix ($ V^T V = I $), columns are **right singular vectors**.

> **Right singular vectors**: Columns of $  V \in \mathbb{R}^{n \times n}  $. These are orthonormal vectors $  v_1, v_2, \dots, v_n \in \mathbb{R}^n  $ in the input (column) space.
>
> **Left singular vectors**: Columns of $  U \in \mathbb{R}^{m \times m}  $. These are orthonormal vectors $  u_1, u_2, \dots, u_m \in \mathbb{R}^m  $ in the output (row) space.

Key properties:
- Singular values are unique and sorted decreasingly.
- Left/right singular vectors are orthonormal bases.

For complex matrices, it's analogous with unitary matrices.

## 2. Geometric Interpretation

SVD reveals how a linear transformation $ A $ acts on space:
- $ V^T $: rotates/reflects the input space to align with principal axes.
- $ \Sigma $: scales along those axes (stretches/compresses by $ \sigma_i $; drops dimensions if $ \sigma_i = 0 $).
- $ U $: rotates the scaled result into the output space.

Think of $ A $ as an ellipse-mapping operator:
- Unit sphere in input space → ellipsoid in output space.
- Right singular vectors = directions of input axes.
- Left singular vectors = directions of output axes.
- Singular values = semi-axis lengths of the ellipsoid.

| Analogy                  | Matrix Part | Meaning                                      |
|--------------------------|-------------|----------------------------------------------|
| Rotation (input)         | $ V^T $   | Aligns input with principal directions       |
| Scaling                  | $ \Sigma $ | Stretches/compresses along those directions  |
| Rotation (output)        | $ U $     | Aligns to output space basis                 |
| Overall                  | $ A $     | Any linear transform = rotation + scale + rotation |

## 3. Connection to Eigenvalues and Other Decompositions

| Decomposition       | Applies To                  | Form                          | Key Insight                              |
|---------------------|-----------------------------|-------------------------------|------------------------------------------|
| Eigendecomposition  | Square, symmetric positive-definite | $ A = Q \Lambda Q^T $       | $ \sigma_i = \sqrt{\lambda_i(A^T A)} $  |
| QR                  | Any matrix                  | $ A = QR $                  | Numerically stable but less revealing    |
| Schur               | Square                      | Upper triangular              | Generalizes eigenvalues                  |
| SVD                 | Any matrix                  | $ A = U \Sigma V^T $        | Most general, reveals rank/energy        |

SVD generalizes eigendecomposition: singular values of $ A $ are square roots of eigenvalues of $ A^T A $ (or $ AA^T $).

## 4. Proof Sketch (Existence)

1. Consider $ A^T A \in \mathbb{R}^{n \times n} $ (symmetric positive semi-definite).
2. By spectral theorem, $ A^T A = V \Lambda V^T $, with $ \Lambda = \operatorname{diag}(\lambda_1 \geq \cdots \geq \lambda_n \geq 0) $.
3. Set $ \sigma_i = \sqrt{\lambda_i} $, $ V $ as right singular vectors.
4. For $ i \leq r $ (rank), define left singular vectors $ u_i = \frac{1}{\sigma_i} A v_i $.
5. Extend $ \{u_i\} $ to orthonormal basis $ U $ for null space (zeros).
6. Verify $ A = U \Sigma V^T $.

This is constructive—most numerical algorithms (Golub-Reinsch) follow similar iterative steps.

## 5. Code Example: Computing and Visualizing SVD

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a random 6x4 matrix (m > n)
np.random.seed(42)
A = np.random.rand(6, 4)

# Compute full SVD
U, sigma, Vt = np.linalg.svd(A, full_matrices=True)

# Sigma is returned as 1D array; reconstruct diagonal matrix
Sigma = np.zeros((6, 4))
np.fill_diagonal(Sigma, sigma)

print("Singular values:", sigma)
print("Rank approximation error (Frobenius):", np.linalg.norm(A - U @ Sigma @ Vt))

# Visualize top 2 singular vectors (for 2D intuition)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].arrow(0, 0, Vt[0, 0], Vt[1, 0], head_width=0.05, color='r', label='v1')
ax[0].arrow(0, 0, Vt[0, 1], Vt[1, 1], head_width=0.05, color='b', label='v2')
ax[0].set_xlim(-1, 1); ax[0].set_ylim(-1, 1)
ax[0].legend(); ax[0].set_title("Right Singular Vectors (Input Space)")

ax[1].arrow(0, 0, U[0, 0], U[1, 0], head_width=0.05, color='r', label='u1')
ax[1].arrow(0, 0, U[0, 1], U[1, 1], head_width=0.05, color='b', label='u2')
ax[1].set_xlim(-1, 1); ax[1].set_ylim(-1, 1)
ax[1].legend(); ax[1].set_title("Left Singular Vectors (Output Space)")

plt.show()
```

**Task for Today (Day 2 of Week 1)**: Run this code in your `attention-svd-studies` repo. Replace the random matrix with a simple 2×2 example (e.g., rotation + scaling) and manually verify the decomposition. Commit the notebook with plots and your written interpretation of what the vectors represent.

## 6. Real-World Applications in Transformers

- **Attention Matrix Analysis**: Attention weights $ A $ (row-stochastic, $ n \times n $) often have rapidly decaying singular values → low effective rank → explains why LoRA (low-rank updates) works.
- **Low-Rank Approximation**: Truncate SVD to k << rank → compress models (e.g., weight matrices in LLMs).
- **Principal Component Analysis (PCA)**: SVD on centered data matrix = PCA.
- **Recommendation Systems**: Matrix factorization for user-item matrices.

Empirical case: In LLaMA-3 technical reports and Grok papers (check latest on arXiv), attention matrices show power-law singular value decay, linking to inducive biases.

## 7. Common Pitfalls and Numerical Issues

- **Numerical stability**: Use `np.linalg.svd(full_matrices=False)` for economy SVD when m >> n.
- **Rank deficiency**: Small singular values ≈ numerical noise; threshold for effective rank.
- **Interpretability**: Top singular vectors often capture global patterns (e.g., positional in early layers, semantic in later).
- **Scaling**: For huge matrices (e.g., in DeepSpeed), use randomized SVD or Lanczos methods.

## 8. Future Implications

SVD underpins emerging work on:
- Dynamic low-rank adaptation (DoRA).
- State-space models (Mamba) bypassing attention via structured matrices with fast SVD-like decompositions.
- Theoretical understanding of why Transformers generalize (singular value spectra link to generalization bounds).