"""A compact educational SigLIP implementation in PyTorch.

This module mirrors the repository's CLIP implementation but swaps the
softmax-based contrastive loss for the pairwise sigmoid loss used by SigLIP.

The implementation intentionally isolates the main conceptual differences:

1. Image and text towers still produce normalized embeddings.
2. Similarities are scaled by a learned logit scale.
3. A learned logit bias shifts pairwise logits.
4. Training uses an elementwise sigmoid loss over all image-text pairs.

This is an educational implementation rather than a checkpoint-faithful replica.
In particular, it uses a simple encoder-style text tower with CLS pooling instead
of reproducing every detail of larger released SigLIP variants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .clip import TransformerEncoder, VisionTransformer, VisionTransformerConfig
except ImportError:  # pragma: no cover - allows direct file execution for smoke tests
    from clip import TransformerEncoder, VisionTransformer, VisionTransformerConfig


@dataclass(frozen=True)
class SigLIPTextConfig:
    """Configuration for the SigLIP text encoder.

    Attributes:
        vocab_size: Token vocabulary size.
        context_length: Maximum text length, excluding the learned CLS token.
        width: Hidden width of the transformer.
        layers: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio in the feed-forward network.
        dropout: Dropout used inside attention and MLP blocks.
        pad_token_id: Token ID used for padding positions.
    """

    vocab_size: int = 32_000
    context_length: int = 64
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0


@dataclass(frozen=True)
class SigLIPConfig:
    """Top-level SigLIP configuration.

    Attributes:
        embed_dim: Shared embedding dimension after modality projections.
        vision_config: ViT configuration for the image tower.
        text_config: Configuration for the text tower.
        logit_scale_init: Initial value of the learned logit-scale parameter.
        logit_bias_init: Initial value of the learned logit bias.
        max_logit_scale: Optional stability clamp before exponentiation.
    """

    embed_dim: int = 512
    vision_config: VisionTransformerConfig = field(default_factory=VisionTransformerConfig)
    text_config: SigLIPTextConfig = field(default_factory=SigLIPTextConfig)
    logit_scale_init: float = math.log(10.0)
    logit_bias_init: float = -10.0
    max_logit_scale: float = math.log(100.0)


@dataclass
class SigLIPOutput:
    """Output container returned by ``SigLIPModel.forward``.

    Attributes:
        image_features: Normalized image embeddings in the shared space.
        text_features: Normalized text embeddings in the shared space.
        logits_per_image: Scaled and shifted image-text logits.
        pairwise_probabilities: Independent pair probabilities after sigmoid.
        loss: Optional pairwise sigmoid loss.
    """

    image_features: Tensor
    text_features: Tensor
    logits_per_image: Tensor
    pairwise_probabilities: Tensor
    loss: Tensor | None = None


class SigLIPTextTransformer(nn.Module):
    """Encoder-style text transformer with CLS pooling.

    Unlike the causal CLIP text tower, this module uses bidirectional
    self-attention and pools the learned CLS token after the encoder stack.
    This keeps the code compact while highlighting a common SigLIP-style setup.
    """

    def __init__(self, config: SigLIPTextConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, config.width) * 0.02)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.context_length + 1, config.width) * 0.01
        )
        self.ln_pre = nn.LayerNorm(config.width)
        self.transformer = TransformerEncoder(
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.ln_post = nn.LayerNorm(config.width)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Encodes token IDs into a pooled text feature.

        Args:
            input_ids: Token IDs with shape ``[batch, seq_len]``.
            attention_mask: Optional padding mask with shape ``[batch, seq_len]``.

        Returns:
            Unprojected text features of shape ``[batch, width]``.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}."
            )

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(
                f"seq_len={seq_len} exceeds context_length={self.config.context_length}."
            )

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        hidden_states = self.token_embedding(input_ids)
        cls_token = self.cls_embedding.expand(batch_size, -1, -1)
        hidden_states = torch.cat([cls_token, hidden_states], dim=1)
        hidden_states = hidden_states + self.position_embedding[:, : seq_len + 1, :]
        hidden_states = self.ln_pre(hidden_states)

        # Prepend a valid position for the CLS token so it can attend to the
        # full sequence and be used as the pooled representation.
        cls_mask = torch.ones(batch_size, 1, device=input_ids.device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        hidden_states = self.transformer(
            hidden_states,
            causal=False,
            attention_mask=full_attention_mask,
        )
        hidden_states = self.ln_post(hidden_states)
        return hidden_states[:, 0]


class SigLIPModel(nn.Module):
    """PyTorch implementation of an educational SigLIP model."""

    def __init__(self, config: SigLIPConfig) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = VisionTransformer(config.vision_config)
        self.text_encoder = SigLIPTextTransformer(config.text_config)

        self.visual_projection = nn.Linear(
            config.vision_config.width,
            config.embed_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            config.text_config.width,
            config.embed_dim,
            bias=False,
        )
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init))
        self.logit_bias = nn.Parameter(torch.tensor(config.logit_bias_init))

    def get_image_features(self, pixel_values: Tensor) -> Tensor:
        """Returns normalized image embeddings in the shared space."""

        image_features = self.vision_encoder(pixel_values)
        image_features = self.visual_projection(image_features)
        return F.normalize(image_features, dim=-1)

    def get_text_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Returns normalized text embeddings in the shared space."""

        text_features = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)

    def compute_pairwise_logits(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        """Computes the scaled and shifted SigLIP pairwise logits."""

        logit_scale = self.logit_scale.clamp(max=self.config.max_logit_scale).exp()
        similarities = torch.matmul(image_features, text_features.T)
        return logit_scale * similarities + self.logit_bias

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_loss: bool = True,
    ) -> SigLIPOutput:
        """Computes SigLIP logits, pairwise probabilities, and optional loss."""

        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask=attention_mask)
        logits_per_image = self.compute_pairwise_logits(image_features, text_features)
        pairwise_probabilities = torch.sigmoid(logits_per_image)

        loss = None
        if return_loss:
            loss = self.compute_siglip_loss(logits_per_image)

        return SigLIPOutput(
            image_features=image_features,
            text_features=text_features,
            logits_per_image=logits_per_image,
            pairwise_probabilities=pairwise_probabilities,
            loss=loss,
        )

    @staticmethod
    def compute_siglip_loss(logits_per_image: Tensor) -> Tensor:
        """Computes the pairwise sigmoid loss used by SigLIP.

        The diagonal entries are positive pairs and all off-diagonal entries are
        treated as negatives. The elementwise labels therefore live in ``{+1, -1}``.
        """

        batch_size, num_texts = logits_per_image.shape
        if batch_size != num_texts:
            raise ValueError(
                "This educational SigLIP loss expects matched image-text batches and "
                "therefore a square similarity matrix."
            )

        labels = torch.full_like(logits_per_image, fill_value=-1.0)
        diagonal_indices = torch.arange(batch_size, device=logits_per_image.device)
        labels[diagonal_indices, diagonal_indices] = 1.0

        # ``softplus(-y * logit)`` is equivalent to ``-log(sigmoid(y * logit))``
        # but tends to be numerically more stable in practical code.
        return F.softplus(-labels * logits_per_image).mean()

    def zero_shot_logits(self, pixel_values: Tensor, class_text_features: Tensor) -> Tensor:
        """Returns scaled and shifted logits for image-text matching."""

        image_features = self.get_image_features(pixel_values)
        if class_text_features.ndim != 2:
            raise ValueError(
                "class_text_features must have shape [num_classes, embed_dim], "
                f"got {tuple(class_text_features.shape)}."
            )

        class_text_features = F.normalize(class_text_features, dim=-1)
        return self.compute_pairwise_logits(image_features, class_text_features)

    def zero_shot_probs(self, pixel_values: Tensor, class_text_features: Tensor) -> Tensor:
        """Returns independent pair probabilities for zero-shot matching.

        SigLIP uses sigmoid rather than softmax because each image-text pair is
        scored independently under the training objective.
        """

        return torch.sigmoid(self.zero_shot_logits(pixel_values, class_text_features))


def build_siglip_tiny() -> SigLIPModel:
    """Builds a very small SigLIP variant for smoke tests and learning."""

    config = SigLIPConfig(
        embed_dim=128,
        vision_config=VisionTransformerConfig(
            image_size=32,
            patch_size=8,
            width=128,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
        ),
        text_config=SigLIPTextConfig(
            vocab_size=1_000,
            context_length=16,
            width=128,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            pad_token_id=0,
        ),
    )
    return SigLIPModel(config)


def _smoke_test() -> None:
    """Runs a tiny forward pass to verify tensor shapes."""

    model = build_siglip_tiny()
    images = torch.randn(4, 3, 32, 32)
    input_ids = torch.tensor(
        [
            [11, 21, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [12, 22, 32, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [13, 23, 33, 43, 53, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [14, 24, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()

    output = model(images, input_ids, attention_mask, return_loss=True)
    print("image_features:", tuple(output.image_features.shape))
    print("text_features:", tuple(output.text_features.shape))
    print("logits_per_image:", tuple(output.logits_per_image.shape))
    print("pairwise_probabilities:", tuple(output.pairwise_probabilities.shape))
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _smoke_test()
