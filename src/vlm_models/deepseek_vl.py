"""A compact educational DeepSeek-VL implementation in PyTorch.

This module follows the architecture summary in ``docs/Large_Models/DeepSeek_VL.md``:

1. A hybrid vision encoder combines a low-resolution semantic branch and a
   high-resolution detail branch.
2. A lightweight vision-language adapter projects fused visual tokens into the
   language model hidden space.
3. A decoder-only language model consumes visual tokens as a prefix and
   predicts text autoregressively.

The code is intentionally educational rather than checkpoint-faithful:

* The low-resolution branch is a ViT-style patch encoder inspired by SigLIP.
* The high-resolution branch is a SAM-inspired CNN detail extractor, not an
  actual SAM-B implementation.
* The language model is a compact causal decoder, not the original DeepSeek-LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .clip import TransformerEncoder, VisionTransformerConfig
except ImportError:  # pragma: no cover - allows direct file execution for smoke tests
    from clip import TransformerEncoder, VisionTransformerConfig


@dataclass(frozen=True)
class HighResVisionConfig:
    """Configuration for the high-resolution detail branch.

    Attributes:
        image_size: Input resolution of the detail branch after resizing.
        in_channels: Number of image channels.
        stem_width: Hidden width used by the convolutional stem.
        output_width: Token width after projection.
        token_grid_size: Spatial grid size used before flattening to tokens.
        dropout: Dropout applied after the projection layer.
    """

    image_size: int = 1024
    in_channels: int = 3
    stem_width: int = 256
    output_width: int = 512
    token_grid_size: int = 24
    dropout: float = 0.0

    @property
    def num_tokens(self) -> int:
        """Returns the number of output tokens in the detail branch."""

        return self.token_grid_size * self.token_grid_size


@dataclass(frozen=True)
class DeepSeekVLLanguageConfig:
    """Configuration for the decoder-only language model.

    Attributes:
        vocab_size: Vocabulary size of the language model.
        context_length: Maximum number of text tokens.
        width: Hidden width of the decoder.
        layers: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio of the feed-forward block.
        dropout: Dropout used in attention and MLP blocks.
        pad_token_id: Padding token ID used for optional attention masking.
    """

    vocab_size: int = 32_000
    context_length: int = 256
    width: int = 1024
    layers: int = 12
    heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0


@dataclass(frozen=True)
class DeepSeekVLConfig:
    """Top-level configuration for the educational DeepSeek-VL model.

    Attributes:
        low_res_vision_config: ViT-like configuration for the semantic branch.
        high_res_vision_config: CNN-style configuration for the detail branch.
        language_config: Configuration for the decoder-only language backbone.
        adapter_hidden_dim: Hidden size used inside the adapter MLPs.
        use_pixel_shuffle_style_resizing: Kept for readability; this
            implementation always uses bilinear resizing internally.
    """

    low_res_vision_config: VisionTransformerConfig = field(
        default_factory=lambda: VisionTransformerConfig(
            image_size=384,
            patch_size=16,
            in_channels=3,
            width=512,
            layers=8,
            heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
        )
    )
    high_res_vision_config: HighResVisionConfig = field(default_factory=HighResVisionConfig)
    language_config: DeepSeekVLLanguageConfig = field(default_factory=DeepSeekVLLanguageConfig)
    adapter_hidden_dim: int = 1024
    use_pixel_shuffle_style_resizing: bool = False


@dataclass
class DeepSeekVLOutput:
    """Output container returned by ``DeepSeekVLModel.forward``.

    Attributes:
        low_res_tokens: Semantic tokens produced by the low-resolution branch.
        high_res_tokens: Detail tokens produced by the high-resolution branch.
        visual_tokens: Adapted visual tokens fed to the language model.
        logits: Next-token logits over the text vocabulary.
        loss: Optional causal language modeling loss.
    """

    low_res_tokens: Tensor
    high_res_tokens: Tensor
    visual_tokens: Tensor
    logits: Tensor
    loss: Tensor | None = None


class PatchTokenVisionTransformer(nn.Module):
    """ViT-style image encoder that returns patch tokens instead of a CLS vector."""

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
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.num_patches, config.width) * 0.02
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
        """Encodes images into a sequence of patch tokens.

        Args:
            pixel_values: Tensor of shape ``[batch, channels, height, width]``.

        Returns:
            Patch tokens with shape ``[batch, num_patches, width]``.
        """

        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width], "
                f"got {tuple(pixel_values.shape)}."
            )

        _, _, height, width = pixel_values.shape
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Expected {self.config.image_size}x{self.config.image_size} input, "
                f"got {height}x{width}."
            )

        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embedding
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = self.transformer(hidden_states, causal=False, attention_mask=None)
        return self.ln_post(hidden_states)


class HighResDetailEncoder(nn.Module):
    """CNN-style high-resolution detail branch inspired by the SAM path.

    The original DeepSeek-VL uses SAM-B features and a carefully designed
    resizing path. This educational version keeps the same intent: preserve
    fine-grained high-resolution detail under a fixed token budget.
    """

    def __init__(self, config: HighResVisionConfig) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.stem_width // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(config.stem_width // 2, config.stem_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.stem_width, config.stem_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.stem_width, config.stem_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Conv2d(config.stem_width, config.output_width, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Encodes high-resolution images into a fixed token grid."""

        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width], "
                f"got {tuple(pixel_values.shape)}."
            )

        _, _, height, width = pixel_values.shape
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Expected {self.config.image_size}x{self.config.image_size} input, "
                f"got {height}x{width}."
            )

        hidden_states = self.stem(pixel_values)
        hidden_states = F.adaptive_avg_pool2d(
            hidden_states,
            output_size=(self.config.token_grid_size, self.config.token_grid_size),
        )
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states.flatten(2).transpose(1, 2)


class HybridVisionEncoder(nn.Module):
    """Combines low-resolution semantic tokens and high-resolution detail tokens."""

    def __init__(
        self,
        low_res_config: VisionTransformerConfig,
        high_res_config: HighResVisionConfig,
    ) -> None:
        super().__init__()
        self.low_res_config = low_res_config
        self.high_res_config = high_res_config
        self.low_res_encoder = PatchTokenVisionTransformer(low_res_config)
        self.high_res_encoder = HighResDetailEncoder(high_res_config)

        if low_res_config.num_patches != high_res_config.num_tokens:
            raise ValueError(
                "Low-resolution patch count and high-resolution token count must match "
                "so the two branches can be fused token-by-token."
            )

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        """Returns tokens from the two vision branches.

        Args:
            pixel_values: Original image tensor. The method resizes it into the
                resolutions required by the low- and high-resolution branches.

        Returns:
            Tuple ``(low_res_tokens, high_res_tokens)`` with matching token counts.
        """

        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width], "
                f"got {tuple(pixel_values.shape)}."
            )

        low_res_pixels = F.interpolate(
            pixel_values,
            size=(self.low_res_config.image_size, self.low_res_config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        high_res_pixels = F.interpolate(
            pixel_values,
            size=(self.high_res_config.image_size, self.high_res_config.image_size),
            mode="bilinear",
            align_corners=False,
        )

        low_res_tokens = self.low_res_encoder(low_res_pixels)
        high_res_tokens = self.high_res_encoder(high_res_pixels)
        return low_res_tokens, high_res_tokens


class VisionLanguageAdapter(nn.Module):
    """Projects hybrid vision features into the language model hidden space."""

    def __init__(
        self,
        low_res_width: int,
        high_res_width: int,
        adapter_hidden_dim: int,
        llm_width: int,
    ) -> None:
        super().__init__()
        self.low_res_adapter = nn.Sequential(
            nn.Linear(low_res_width, adapter_hidden_dim),
            nn.GELU(),
        )
        self.high_res_adapter = nn.Sequential(
            nn.Linear(high_res_width, adapter_hidden_dim),
            nn.GELU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(adapter_hidden_dim * 2, llm_width),
            nn.GELU(),
            nn.Linear(llm_width, llm_width),
        )

    def forward(self, low_res_tokens: Tensor, high_res_tokens: Tensor) -> Tensor:
        """Fuses and projects semantic/detail tokens to the LM hidden width."""

        if low_res_tokens.shape[:2] != high_res_tokens.shape[:2]:
            raise ValueError(
                "Low-resolution and high-resolution branches must produce the same "
                "batch size and token count."
            )

        low_res_tokens = self.low_res_adapter(low_res_tokens)
        high_res_tokens = self.high_res_adapter(high_res_tokens)
        fused_tokens = torch.cat([low_res_tokens, high_res_tokens], dim=-1)
        return self.fusion_mlp(fused_tokens)


class DeepSeekVLLanguageModel(nn.Module):
    """A compact decoder-only language model with visual prefix support.

    This module is intentionally small and readable. It uses a standard causal
    transformer over the concatenation ``[visual_prefix, text_tokens]``. The
    visual prefix acts like a learned multimodal prompt for the decoder.
    """

    def __init__(self, config: DeepSeekVLLanguageConfig, max_visual_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.max_visual_tokens = max_visual_tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_visual_tokens + config.context_length, config.width) * 0.01
        )
        self.transformer = TransformerEncoder(
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )
        self.ln_final = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        visual_tokens: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Runs causal decoding over visual tokens plus text tokens.

        Args:
            input_ids: Text token IDs of shape ``[batch, text_len]``.
            visual_tokens: Adapted visual tokens of shape ``[batch, visual_len, width]``.
            attention_mask: Optional text padding mask of shape ``[batch, text_len]``.
            labels: Optional labels for next-token prediction. Use ``-100`` for
                positions that should be ignored by the loss.

        Returns:
            Tuple ``(logits, loss)`` where logits have shape
            ``[batch, text_len, vocab_size]``.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, text_len], got {tuple(input_ids.shape)}."
            )
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [batch, visual_len, hidden], "
                f"got {tuple(visual_tokens.shape)}."
            )

        batch_size, text_len = input_ids.shape
        _, visual_len, hidden_width = visual_tokens.shape
        if hidden_width != self.config.width:
            raise ValueError(
                f"visual token width {hidden_width} does not match language width {self.config.width}."
            )
        if visual_len > self.max_visual_tokens:
            raise ValueError(
                f"visual_len={visual_len} exceeds max_visual_tokens={self.max_visual_tokens}."
            )
        if text_len > self.config.context_length:
            raise ValueError(
                f"text_len={text_len} exceeds context_length={self.config.context_length}."
            )

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        text_embeddings = self.token_embedding(input_ids)
        hidden_states = torch.cat([visual_tokens, text_embeddings], dim=1)
        hidden_states = hidden_states + self.position_embedding[:, : visual_len + text_len, :]

        # All visual prefix tokens are considered valid context. Text padding is
        # appended afterward so that the decoder can ignore padded text positions.
        visual_mask = torch.ones(
            batch_size,
            visual_len,
            device=input_ids.device,
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        hidden_states = self.transformer(
            hidden_states,
            causal=True,
            attention_mask=full_attention_mask,
        )
        hidden_states = self.ln_final(hidden_states)

        # Only text positions are projected to vocabulary logits. Visual prefix
        # tokens serve as context and are not themselves decoded as words.
        text_hidden_states = hidden_states[:, visual_len:, :]
        logits = self.lm_head(text_hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels must have shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}."
                )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss


class DeepSeekVLModel(nn.Module):
    """Educational implementation of the DeepSeek-VL architecture."""

    def __init__(self, config: DeepSeekVLConfig) -> None:
        super().__init__()
        self.config = config
        self.hybrid_vision_encoder = HybridVisionEncoder(
            config.low_res_vision_config,
            config.high_res_vision_config,
        )
        self.adapter = VisionLanguageAdapter(
            low_res_width=config.low_res_vision_config.width,
            high_res_width=config.high_res_vision_config.output_width,
            adapter_hidden_dim=config.adapter_hidden_dim,
            llm_width=config.language_config.width,
        )
        self.language_model = DeepSeekVLLanguageModel(
            config.language_config,
            max_visual_tokens=config.low_res_vision_config.num_patches,
        )

    def encode_image(self, pixel_values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encodes images and returns branch tokens plus adapted visual tokens."""

        low_res_tokens, high_res_tokens = self.hybrid_vision_encoder(pixel_values)
        visual_tokens = self.adapter(low_res_tokens, high_res_tokens)
        return low_res_tokens, high_res_tokens, visual_tokens

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> DeepSeekVLOutput:
        """Runs multimodal decoding on image and text inputs.

        Args:
            pixel_values: Image tensor of shape ``[batch, channels, height, width]``.
            input_ids: Text token IDs of shape ``[batch, text_len]``.
            attention_mask: Optional text padding mask.
            labels: Optional labels for next-token prediction.

        Returns:
            ``DeepSeekVLOutput`` containing branch tokens, fused visual tokens,
            text logits, and an optional loss.
        """

        low_res_tokens, high_res_tokens, visual_tokens = self.encode_image(pixel_values)
        logits, loss = self.language_model(
            input_ids=input_ids,
            visual_tokens=visual_tokens,
            attention_mask=attention_mask,
            labels=labels,
        )
        return DeepSeekVLOutput(
            low_res_tokens=low_res_tokens,
            high_res_tokens=high_res_tokens,
            visual_tokens=visual_tokens,
            logits=logits,
            loss=loss,
        )


def build_deepseek_vl_tiny() -> DeepSeekVLModel:
    """Builds a very small DeepSeek-VL variant for smoke tests and study."""

    config = DeepSeekVLConfig(
        low_res_vision_config=VisionTransformerConfig(
            image_size=32,
            patch_size=8,
            in_channels=3,
            width=96,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
        ),
        high_res_vision_config=HighResVisionConfig(
            image_size=64,
            in_channels=3,
            stem_width=64,
            output_width=96,
            token_grid_size=4,
            dropout=0.0,
        ),
        language_config=DeepSeekVLLanguageConfig(
            vocab_size=512,
            context_length=24,
            width=128,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
        ),
        adapter_hidden_dim=128,
    )
    return DeepSeekVLModel(config)


def _smoke_test() -> None:
    """Runs a small forward pass to verify tensor shapes."""

    model = build_deepseek_vl_tiny()
    images = torch.randn(2, 3, 48, 48)
    input_ids = torch.tensor(
        [
            [11, 21, 31, 41, 51, 0, 0, 0],
            [12, 22, 32, 42, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()
    labels = input_ids.masked_fill(input_ids == 0, -100)

    output = model(
        pixel_values=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    print("low_res_tokens:", tuple(output.low_res_tokens.shape))
    print("high_res_tokens:", tuple(output.high_res_tokens.shape))
    print("visual_tokens:", tuple(output.visual_tokens.shape))
    print("logits:", tuple(output.logits.shape))
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _smoke_test()
