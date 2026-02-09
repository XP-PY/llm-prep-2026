# [SigLIP](https://arxiv.org/abs/2303.15343)

**SigLIP** (Sigmoid Loss for Language Image Pre-Training) is a family of vision-language models developed by Google DeepMind in 2023. It builds directly on the contrastive image-text pre-training paradigm introduced by CLIP (2021) but replaces the standard softmax-based contrastive loss with a **pairwise sigmoid loss**. This seemingly small change yields significant practical advantages in training efficiency, scalability, and performance, especially at smaller batch sizes or with limited compute.

## 1. Core Idea & Motivation

- **Goal**: Learn aligned image and text embeddings from noisy web-scale image-text pairs (no class labels needed) for strong **zero-shot** transfer.
- **Problem with CLIP-style softmax loss**:
  - Requires global normalization over the entire batch → expensive all-gather in distributed training.
  - Memory-intensive for large batches.
  - Numerically tricky (needs log-sum-exp stabilization).
  - Performance degrades noticeably at batch sizes < ~16k.
- **SigLIP solution**: Treat every image-text pair in the batch independently as a binary classification problem using sigmoid + binary cross-entropy.
  - No global view needed → simple, memory-efficient, enables extreme batch sizes (up to 1M demonstrated).
  - Symmetric by design (no need for separate image-to-text and text-to-image passes).
  - Works better at small batches and converges similarly at large ones.

Two variants introduced:
- **SigLIP**: Trains both image and text encoders from scratch (like CLIP).
- **SigLiT** (Locked-image Text tuning): Freezes a pre-trained image encoder (e.g., ViT) and only trains the text tower → extremely compute-efficient.

## 2. Key Advantages Over CLIP

| Aspect                  | CLIP (Softmax)                          | SigLIP (Sigmoid)                               |
|-------------------------|-----------------------------------------|------------------------------------------------|
| Loss                    | InfoNCE + softmax (relative ranking)    | Pairwise BCE (absolute binary decisions)       |
| Batch size scaling      | Limited by all-gather (~32k practical)  | Up to 1M (chunked implementation)              |
| Small-batch performance | Weaker                                  | Significantly stronger                         |
| Distributed efficiency  | High communication cost                 | Low (no all-gather needed)                     |
| Symmetry                | Explicit dual directions                | Natural single loss                            |
| Extra parameters        | Temperature τ                           | Temperature τ + bias b                         |
| Zero-shot ImageNet      | ~63–76% (original)                      | Up to 84.5% (SigLiT-g/14 in 2 days on 4 TPUs)  |
| Training cost example   | ~10 days on 256 TPUv3                   | 84.5% in 2 days on 4 TPUv4                      |

## 3. Architecture

- **Image encoder**: Vision Transformer (ViT) variants (B/16, L/16, g/14, etc.), same as CLIP.
  - Patch embedding via Conv2d.
  - No major changes in backbone.
- **Text encoder**: Transformer (often smaller than image tower).
- **Pooling**:
  - Early/small models: Standard **[CLS] token** (like CLIP).
  - Larger/recent models (e.g., `siglip-so400m-patch14-384`): **Multihead Attention Pooling (MAP)** head with a learnable probe/query (more dynamic aggregation).
- **Final projection**: Linear layer to shared embedding space + L2 normalization.

## 4. Loss Function (Mathematical Details)

For batch size B, embeddings normalized:

- Similarity matrix: \( s_{ij} = x_i \cdot y_j \)
- Logit: \( l_{ij} = t \cdot s_{ij} + b \)  (t = exp(t′), b learned)
- Label matrix: \( z_{ij} = +1 \) if i==j else -1
- Loss:
  \[
  L = -\frac{1}{B^2} \sum_{i,j} \log \sigma \left( z_{ij} \cdot l_{ij} \right)
  \]

Initializations:
- t′ = log(10) → t ≈ 10
- b = -10 (helps with initial positive/negative imbalance)

## 5. PyTorch Implementation Highlights (Hugging Face)

```python
# Zero-shot classification (same as CLIP)
from transformers import AutoProcessor, AutoModel
from PIL import Image

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

image_emb = outputs.image_embeds  # [1, dim] (pooled, normalized)
text_embs = outputs.text_embeds   # [num_texts, dim]

logits = image_emb @ text_embs.T  # cosine similarities
probs = logits.softmax(dim=-1)
```

Key internal components:
- Vision embeddings → same as CLIP (patch conv + optional interpolated pos encoding).
- Pooling head (in larger models):
  ```python
  class SiglipMultiheadAttentionPoolingHead(nn.Module):
      def __init__(self, config):
          self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
          self.attention = nn.MultiheadAttention(config.hidden_size,
                                                 config.num_attention_heads,
                                                 batch_first=True)
          self.layernorm = nn.LayerNorm(config.hidden_size)
          self.mlp = SiglipMLP(config)  # small FFN

      def forward(self, hidden_state):  # [B, seq_len, dim]
          probe = self.probe.repeat(hidden_state.shape[0], 1, 1)
          attended = self.attention(probe, hidden_state, hidden_state)[0]  # [B, 1, dim]
          residual = attended
          attended = self.layernorm(attended)
          attended = residual + self.mlp(attended)
          return attended[:, 0]  # [B, dim]
  ```

## 6. Notable Results (from Paper)

| Model          | Image Encoder | Text Encoder | Batch Size | TPUs | Days | ImageNet 0-shot |
|----------------|---------------|--------------|------------|------|------|-----------------|
| SigLiT         | frozen g/14   | L            | 20k        | 4    | 2    | **84.5%**       |
| SigLiT         | frozen B/8    | L*           | 32k        | 4    | 1    | 79.8%           |
| SigLIP         | B/16          | B            | 32k        | 32   | 5    | 73.4%           |

- Batch size saturation: 32k is sufficient; 1M gives negligible gains.
- Multilingual training also benefits.

## 7. Available Models (Hugging Face, as of early 2026)

- `google/siglip-base-patch16-224`
- `google/siglip-large-patch16-384`
- `google/siglip-so400m-patch14-384` (400M image encoder, strong performance)
- `google/siglip2-*` variants (newer iterations with improvements)

## 8. Summary & Why It Matters

SigLIP is a refined, more practical evolution of CLIP:
- Same zero-shot capabilities.
- Dramatically lower training cost.
- Better scaling behavior.
- Drop-in replacement in most code (Hugging Face API identical to CLIP).

For VLM practitioners, SigLIP models are now often the default choice for strong open-source vision-language embeddings due to their efficiency and performance.