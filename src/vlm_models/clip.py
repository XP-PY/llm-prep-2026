"""A compact educational CLIP implementation in PyTorch.

This module focuses on the core ideas described in ``docs/Large_Models/CLIP.md``:

1. An image encoder and a text encoder produce modality-specific features.
2. Separate linear projections map both features into a shared embedding space.
3. The final embeddings are L2-normalized, so cosine similarity becomes a dot product.
4. A learnable temperature rescales similarities before the symmetric contrastive loss.

The implementation intentionally keeps the training pipeline simple:

* The image encoder is a ViT-style patch transformer.
* The text encoder is a causal transformer with EOS pooling.
* Tokenization and image preprocessing are left to the caller because they vary by project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class VisionTransformerConfig:
    """Configuration for the ViT image encoder used inside CLIP.

    Attributes:
        image_size: Input image resolution expected by the patch embedding layer.
        patch_size: Spatial size of each patch.
        in_channels: Number of image channels. RGB images use 3.
        width: Hidden width of the transformer.
        layers: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio in the feed-forward block.
        dropout: Dropout applied inside attention and MLP blocks.
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    width: int = 768
    layers: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        """Returns the number of spatial tokens produced by patchification."""
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side


@dataclass(frozen=True)
class TextTransformerConfig:
    """Configuration for the causal text encoder used inside CLIP.

    Attributes:
        vocab_size: Token vocabulary size.
        context_length: Maximum number of tokens, including special tokens.
        width: Hidden width of the transformer.
        layers: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio in the feed-forward block.
        dropout: Dropout applied inside attention and MLP blocks.
        pad_token_id: Token ID used for padding.
        eos_token_id: Token ID used for end-of-sequence pooling.
    """

    vocab_size: int = 49_152
    context_length: int = 77
    width: int = 512
    layers: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0
    eos_token_id: int = 2


@dataclass(frozen=True)
class CLIPConfig:
    """Top-level CLIP configuration.

    Attributes:
        embed_dim: Shared multimodal embedding dimension after projection.
        vision_config: Configuration for the image encoder.
        text_config: Configuration for the text encoder.
        logit_scale_init: Initial value for the log temperature parameter.
        max_logit_scale: Safety clamp used before exponentiation.
    """

    embed_dim: int = 512
    vision_config: VisionTransformerConfig = field(default_factory=VisionTransformerConfig)
    text_config: TextTransformerConfig = field(default_factory=TextTransformerConfig)
    logit_scale_init: float = math.log(1 / 0.07)
    max_logit_scale: float = math.log(100.0)


@dataclass
class CLIPOutput:
    """Output container returned by ``CLIPModel.forward``.

    Attributes:
        image_features: Normalized image embeddings in the shared space.
        text_features: Normalized text embeddings in the shared space.
        logits_per_image: Similarity matrix for image-to-text retrieval.
        logits_per_text: Similarity matrix for text-to-image retrieval.
        loss: Optional symmetric contrastive loss.
    """

    image_features: Tensor
    text_features: Tensor
    logits_per_image: Tensor
    logits_per_text: Tensor
    loss: Tensor | None = None


class QuickGELU(nn.Module):
    """Applies the QuickGELU activation used by CLIP-like models."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * torch.sigmoid(1.702 * inputs)


class MultiHeadSelfAttention(nn.Module):
    """Minimal multi-head self-attention block.

    This implementation is deliberately explicit instead of heavily optimized.
    The goal is to make tensor shapes and masking behavior easy to inspect when
    revisiting CLIP or transformer papers later.
    """

    def __init__(self, width: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")

        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(width, width * 3)
        self.out_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Runs self-attention over a sequence.

        Args:
            hidden_states: Input sequence of shape ``[batch, seq_len, width]``.
            causal: Whether to apply an upper-triangular causal mask.
            attention_mask: Optional padding mask of shape ``[batch, seq_len]``
                where 1 means a valid token and 0 means padding.

        Returns:
            Updated hidden states with the same shape as the input.
        """

        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, and V in one projection because the three tensors start
        # from the same hidden states and differ only by learned linear maps.
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=1,
            )
            min_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(causal_mask, min_value)

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have shape "
                    f"[batch, seq_len], got {tuple(attention_mask.shape)}."
                )

            # Only keys are masked here. Query-side padded positions are harmless
            # because CLIP later pools the EOS token rather than every token.
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            min_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(~key_mask, min_value)

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.width)
        return self.out_proj(context)


class MLP(nn.Module):
    """Transformer feed-forward block with QuickGELU."""

    def __init__(self, width: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden_dim)
        self.act = QuickGELU()
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.dropout(hidden_states)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block used by both encoders."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = MultiHeadSelfAttention(width, num_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = MLP(width, mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        # Residual connections are the mechanism that lets very deep
        # transformers preserve a stable information path during optimization.
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states),
            causal=causal,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states


class TransformerEncoder(nn.Module):
    """Stack of transformer blocks."""

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    width=width,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        *,
        causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        for block in self.layers:
            hidden_states = block(
                hidden_states,
                causal=causal,
                attention_mask=attention_mask,
            )
        return hidden_states


class VisionTransformer(nn.Module):
    """ViT-style image encoder used by this CLIP implementation."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.config = config
        self.patch_embed = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.width) * 0.02)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.width) * 0.02
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

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Encodes a batch of images into global visual features.

        Args:
            pixel_values: Image tensor of shape ``[batch, channels, height, width]``.

        Returns:
            Unprojected visual features of shape ``[batch, width]``.
        """

        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width], "
                f"got {tuple(pixel_values.shape)}."
            )

        _, _, height, width = pixel_values.shape
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                "This educational implementation expects fixed-size square inputs. "
                f"Expected {self.config.image_size}x{self.config.image_size}, "
                f"got {height}x{width}."
            )

        # Convert the image into a grid of patch tokens. Each output location
        # corresponds to one non-overlapping image patch.
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        hidden_states = torch.cat([class_token, hidden_states], dim=1)

        hidden_states = hidden_states + self.position_embedding
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = self.transformer(hidden_states, causal=False, attention_mask=None)
        hidden_states = self.ln_post(hidden_states)

        # The class token aggregates global image information after self-attention.
        return hidden_states[:, 0]


class TextTransformer(nn.Module):
    """Causal text encoder used by this CLIP implementation."""

    def __init__(self, config: TextTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.context_length, config.width) * 0.01
        )
        self.transformer = TransformerEncoder(
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.ln_final = nn.LayerNorm(config.width)

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
        hidden_states = hidden_states + self.position_embedding[:, :seq_len, :]
        hidden_states = self.transformer(
            hidden_states,
            causal=True,
            attention_mask=attention_mask,
        )
        hidden_states = self.ln_final(hidden_states)

        pooled_indices = self._find_eos_positions(input_ids, attention_mask)
        pooled_output = hidden_states[torch.arange(batch_size, device=input_ids.device), pooled_indices]
        return pooled_output

    def _find_eos_positions(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Finds the pooling position for each sequence.

        CLIP commonly uses the hidden state at the EOS token because that token
        has attended to the full prefix under the causal mask. If EOS is absent,
        we fall back to the last non-padding token to keep the module robust
        during debugging or synthetic smoke tests.
        """

        eos_mask = input_ids.eq(self.config.eos_token_id)
        has_eos = eos_mask.any(dim=-1)

        # ``argmax`` returns the first True position when applied to an integer
        # mask, which matches the pooling strategy described in many CLIP codebases.
        eos_positions = eos_mask.to(torch.int64).argmax(dim=-1)
        last_non_pad = attention_mask.to(torch.int64).sum(dim=-1) - 1
        last_non_pad = last_non_pad.clamp_min(0)
        return torch.where(has_eos, eos_positions, last_non_pad)


class CLIPModel(nn.Module):
    """PyTorch implementation of a ViT-based CLIP model."""

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = VisionTransformer(config.vision_config)
        self.text_encoder = TextTransformer(config.text_config)

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

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_loss: bool = True,
    ) -> CLIPOutput:
        """Computes CLIP similarities and an optional contrastive loss.

        Args:
            pixel_values: Input image tensor.
            input_ids: Tokenized text tensor.
            attention_mask: Optional mask for padded tokens.
            return_loss: Whether to compute the symmetric CLIP loss.

        Returns:
            A ``CLIPOutput`` object containing embeddings, logits, and loss.
        """

        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask=attention_mask)

        # The learnable temperature is stored in log space for stability and then
        # exponentiated before similarity scaling.
        logit_scale = self.logit_scale.clamp(max=self.config.max_logit_scale).exp()
        logits_per_image = torch.matmul(image_features, text_features.T) * logit_scale
        logits_per_text = logits_per_image.T

        loss = None
        if return_loss:
            loss = self.compute_clip_loss(logits_per_image, logits_per_text)

        return CLIPOutput(
            image_features=image_features,
            text_features=text_features,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            loss=loss,
        )

    @staticmethod
    def compute_clip_loss(logits_per_image: Tensor, logits_per_text: Tensor) -> Tensor:
        """Computes the symmetric InfoNCE-style CLIP loss."""

        if logits_per_image.shape[0] != logits_per_image.shape[1]:
            raise ValueError(
                "CLIP loss expects a square similarity matrix built from matched image-text pairs."
            )

        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2

    def zero_shot_logits(self, pixel_values: Tensor, class_text_features: Tensor) -> Tensor:
        """Returns image-to-class similarity scores for zero-shot classification.

        Args:
            pixel_values: Batch of images.
            class_text_features: Precomputed normalized text embeddings for prompts
                such as ``"a photo of a dog."`` with shape ``[num_classes, embed_dim]``.

        Returns:
            Similarity logits of shape ``[batch, num_classes]``.
        """

        image_features = self.get_image_features(pixel_values)
        if class_text_features.ndim != 2:
            raise ValueError(
                "class_text_features must have shape [num_classes, embed_dim], "
                f"got {tuple(class_text_features.shape)}."
            )

        class_text_features = F.normalize(class_text_features, dim=-1)
        logit_scale = self.logit_scale.clamp(max=self.config.max_logit_scale).exp()
        return torch.matmul(image_features, class_text_features.T) * logit_scale


def build_clip_tiny() -> CLIPModel:
    """Builds a very small CLIP variant for smoke tests and learning."""

    config = CLIPConfig(
        embed_dim=128,
        vision_config=VisionTransformerConfig(
            image_size=32,
            patch_size=8,
            width=128,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
        ),
        text_config=TextTransformerConfig(
            vocab_size=1_000,
            context_length=16,
            width=128,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            pad_token_id=0,
            eos_token_id=2,
        ),
    )
    return CLIPModel(config)


def _smoke_test() -> None:
    """Runs a tiny forward pass to verify tensor shapes."""

    model = build_clip_tiny()
    images = torch.randn(4, 3, 32, 32)
    input_ids = torch.tensor(
        [
            [1, 11, 21, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 12, 22, 32, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 13, 23, 33, 43, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 14, 24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()

    output = model(images, input_ids, attention_mask, return_loss=True)
    print("image_features:", tuple(output.image_features.shape))
    print("text_features:", tuple(output.text_features.shape))
    print("logits_per_image:", tuple(output.logits_per_image.shape))
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _smoke_test()
