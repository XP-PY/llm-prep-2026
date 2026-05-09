"""A compact educational Gemma 4 implementation in PyTorch.

This module focuses on the architectural ideas that define the Gemma 4 E2B and
E4B releases described in ``docs/Large_Models/Gemma_4.md``:

1. A multimodal prefix built from text, image, and audio inputs.
2. A variable-resolution vision tower with 3x3 pooling into soft visual tokens.
3. A lightweight audio tower for the E2B / E4B small-model path.
4. A decoder-only language model with hybrid local/global attention.
5. Per-Layer Embeddings (PLE) that inject a side-channel into every decoder
   block, matching the high-level mechanism described in the official docs.

This is intentionally educational rather than checkpoint-faithful:

* The exact production p-RoPE recipe is approximated with standard RoPE bases.
* The production KV-sharing optimization is not reproduced layer-for-layer.
* The PLE token-ID path uses a hashed side table so the implementation remains
  lightweight enough for study and smoke tests.
* The audio tower is a conformer-like approximation rather than Google's exact
  USM implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .clip import MultiHeadSelfAttention
except ImportError:  # pragma: no cover - allows direct file execution for smoke tests
    from clip import MultiHeadSelfAttention


def repeat_attention_pattern(num_sliding_layers: int, num_cycles: int) -> tuple[str, ...]:
    """Builds a Gemma 4 style local/global pattern.

    E2B uses ``4 sliding + 1 full`` repeated seven times, while E4B uses
    ``5 sliding + 1 full`` repeated seven times. Tiny builders below keep the
    same pattern but with fewer cycles.
    """

    if num_sliding_layers < 1:
        raise ValueError("num_sliding_layers must be at least 1.")
    if num_cycles < 1:
        raise ValueError("num_cycles must be at least 1.")

    pattern: list[str] = []
    for _ in range(num_cycles):
        pattern.extend(["sliding_attention"] * num_sliding_layers)
        pattern.append("full_attention")
    return tuple(pattern)


@dataclass(frozen=True)
class Gemma4VisionConfig:
    """Configuration for the educational Gemma 4 vision stack.

    Attributes:
        patch_size: Patch size used by the vision patch embedding stem.
        pool_kernel_size: Pooling kernel used before soft-token compression.
        soft_tokens_per_image: Default number of visual tokens per image.
        allowed_soft_token_budgets: Supported visual token budgets.
        width: Hidden width of the ViT-like vision stack.
        layers: Number of vision transformer layers.
        heads: Attention heads in the vision transformer.
        mlp_ratio: FFN expansion ratio inside the vision transformer.
        dropout: Dropout used inside the vision transformer.
        max_axis_positions: Maximum number of row/column indices supported by
            the learned 2D positional embeddings.
    """

    patch_size: int = 16
    pool_kernel_size: int = 3
    soft_tokens_per_image: int = 280
    allowed_soft_token_budgets: tuple[int, ...] = (70, 140, 280, 560, 1120)
    width: int = 192
    layers: int = 4
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_axis_positions: int = 512


@dataclass(frozen=True)
class Gemma4AudioConfig:
    """Configuration for the educational Gemma 4 audio stack.

    Attributes:
        feature_dim: Feature size of each audio timestep. In practice this can
            represent log-mel bins or any other precomputed frame feature.
        width: Hidden width used by the conformer-like audio stack.
        layers: Number of audio encoder layers.
        heads: Attention heads in the audio stack.
        mlp_ratio: FFN expansion ratio inside the audio stack.
        dropout: Dropout used in the audio stack.
        conv_kernel_size: Kernel size of the depthwise convolution module.
        compress_tokens_per_clip: Number of fixed audio tokens output per clip.
        max_positions: Maximum temporal positions for sinusoidal embeddings.
    """

    feature_dim: int = 80
    width: int = 256
    layers: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    conv_kernel_size: int = 5
    compress_tokens_per_clip: int = 64
    max_positions: int = 4096


@dataclass(frozen=True)
class Gemma4TextConfig:
    """Configuration for the Gemma 4 decoder backbone.

    Attributes:
        vocab_size: Text vocabulary size.
        context_length: Maximum text context length.
        width: Hidden width of the decoder.
        layers: Number of decoder blocks.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads used in local attention.
        head_dim: Per-head width used in sliding-window layers.
        global_head_dim: Per-head width used in global layers.
        local_window_size: Sliding-window span for local layers.
        layer_types: Tuple describing whether each layer uses local or global
            attention.
        mlp_ratio: FFN expansion ratio.
        dropout: Dropout used in attention and FFN blocks.
        pad_token_id: Padding token ID for causal language modeling.
        local_rope_base: RoPE base used in sliding-window layers.
        global_rope_base: RoPE base used in global layers.
        hidden_size_per_layer_input: Width of the compact PLE side-channel.
        ple_hash_buckets: Number of hash buckets used for the compact token-ID
            path inside the educational PLE implementation.
        num_kv_shared_layers: Metadata copied from the official configs. The
            educational implementation keeps this value for transparency but
            does not exactly reproduce the production KV-sharing scheme.
    """

    vocab_size: int = 4096
    context_length: int = 256
    width: int = 256
    layers: int = 10
    num_query_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int = 64
    global_head_dim: int = 128
    local_window_size: int = 16
    layer_types: tuple[str, ...] = field(default_factory=lambda: repeat_attention_pattern(4, 2))
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0
    local_rope_base: float = 10_000.0
    global_rope_base: float = 1_000_000.0
    hidden_size_per_layer_input: int = 64
    ple_hash_buckets: int = 2048
    num_kv_shared_layers: int = 0


@dataclass(frozen=True)
class Gemma4Config:
    """Top-level configuration for the educational Gemma 4 implementation."""

    variant_name: str = "Gemma4-E2B-educational"
    supports_audio: bool = True
    vision_config: Gemma4VisionConfig = field(default_factory=Gemma4VisionConfig)
    audio_config: Gemma4AudioConfig = field(default_factory=Gemma4AudioConfig)
    text_config: Gemma4TextConfig = field(default_factory=Gemma4TextConfig)


@dataclass
class Gemma4Output:
    """Output container returned by ``Gemma4Model.forward``.

    Attributes:
        visual_tokens: Padded visual prefix tokens.
        visual_attention_mask: Valid-token mask for the visual prefix.
        audio_tokens: Padded audio prefix tokens.
        audio_attention_mask: Valid-token mask for the audio prefix.
        logits: Text next-token logits.
        loss: Optional causal language modeling loss.
        chosen_patch_grids: Patch-grid choices for each image in each sample.
        num_images_per_sample: Number of images used per sample.
        audio_token_lengths: Number of valid audio tokens per sample.
    """

    visual_tokens: Tensor
    visual_attention_mask: Tensor
    audio_tokens: Tensor
    audio_attention_mask: Tensor
    logits: Tensor
    loss: Tensor | None
    chosen_patch_grids: list[list[tuple[int, int]]]
    num_images_per_sample: list[int]
    audio_token_lengths: list[int]


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by Gemma-family models."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


def rotate_half(hidden_states: Tensor) -> Tensor:
    """Applies the standard half-rotation used inside RoPE."""

    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat([-second_half, first_half], dim=-1)


def build_rope_cache(
    sequence_length: int,
    head_dim: int,
    *,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Builds cosine and sine caches for rotary positional embeddings."""

    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")

    half_dim = head_dim // 2
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    frequency_exponents = torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
    inverse_frequencies = base ** (-frequency_exponents)
    angles = torch.outer(positions, inverse_frequencies)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1).to(dtype)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def apply_rope(hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Applies RoPE to a ``[batch, heads, seq_len, head_dim]`` tensor."""

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return hidden_states * cos + rotate_half(hidden_states) * sin


def build_sinusoidal_positions(
    sequence_length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Returns standard sinusoidal position encodings."""

    if width % 2 != 0:
        raise ValueError(f"Sinusoidal position width must be even, got {width}.")

    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    div_terms = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    sinusoid = torch.zeros(sequence_length, width, device=device, dtype=torch.float32)
    sinusoid[:, 0::2] = torch.sin(positions[:, None] * div_terms[None, :])
    sinusoid[:, 1::2] = torch.cos(positions[:, None] * div_terms[None, :])
    return sinusoid.to(dtype)


def normalize_image_batch(pixel_values: Tensor | list[Tensor] | list[list[Tensor]]) -> list[list[Tensor]]:
    """Normalizes multiple image input formats into ``list[list[Tensor]]``.

    Supported inputs:

    * ``Tensor[batch, channels, height, width]``
    * ``Tensor[batch, num_images, channels, height, width]``
    * ``list[Tensor[channels, height, width]]``
    * ``list[list[Tensor[channels, height, width]]]``
    """

    if isinstance(pixel_values, Tensor):
        if pixel_values.ndim == 4:
            return [[image] for image in pixel_values]
        if pixel_values.ndim == 5:
            return [[image for image in sample] for sample in pixel_values]
        raise ValueError(
            "pixel_values tensor must have shape [batch, channels, height, width] or "
            f"[batch, num_images, channels, height, width], got {tuple(pixel_values.shape)}."
        )

    samples = list(pixel_values)
    if not samples:
        raise ValueError("pixel_values cannot be empty.")

    normalized_samples: list[list[Tensor]] = []
    for sample in samples:
        if isinstance(sample, Tensor):
            if sample.ndim == 3:
                normalized_samples.append([sample])
            elif sample.ndim == 4:
                normalized_samples.append([image for image in sample])
            else:
                raise ValueError(
                    "Each tensor sample must have shape [channels, height, width] or "
                    f"[num_images, channels, height, width], got {tuple(sample.shape)}."
                )
            continue

        image_list = list(sample)
        if not image_list:
            raise ValueError("Nested image sample cannot be empty.")
        for image in image_list:
            if image.ndim != 3:
                raise ValueError(
                    f"Each image must have shape [channels, height, width], got {tuple(image.shape)}."
                )
        normalized_samples.append(image_list)

    return normalized_samples


def normalize_audio_batch(audio_values: Tensor | list[Tensor]) -> list[list[Tensor]]:
    """Normalizes audio input into ``list[list[Tensor]]``.

    Supported inputs:

    * ``Tensor[batch, time, feature_dim]``
    * ``Tensor[batch, num_clips, time, feature_dim]``
    * ``list[Tensor[time, feature_dim]]``
    * ``list[list[Tensor[time, feature_dim]]]``
    """

    if isinstance(audio_values, Tensor):
        if audio_values.ndim == 3:
            return [[clip] for clip in audio_values]
        if audio_values.ndim == 4:
            return [[clip for clip in sample] for sample in audio_values]
        raise ValueError(
            "audio_values tensor must have shape [batch, time, feature_dim] or "
            f"[batch, num_clips, time, feature_dim], got {tuple(audio_values.shape)}."
        )

    samples = list(audio_values)
    if not samples:
        raise ValueError("audio_values cannot be empty.")

    normalized_samples: list[list[Tensor]] = []
    for sample in samples:
        if isinstance(sample, Tensor):
            if sample.ndim != 2:
                raise ValueError(
                    f"Each audio clip tensor must have shape [time, feature_dim], got {tuple(sample.shape)}."
                )
            normalized_samples.append([sample])
            continue

        clip_list = list(sample)
        if not clip_list:
            raise ValueError("Nested audio sample cannot be empty.")
        for clip in clip_list:
            if clip.ndim != 2:
                raise ValueError(
                    f"Each audio clip must have shape [time, feature_dim], got {tuple(clip.shape)}."
                )
        normalized_samples.append(clip_list)

    return normalized_samples


class FeedForward(nn.Module):
    """FFN block using the GELU-tanh activation from the official configs."""

    def __init__(self, width: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.dropout(hidden_states)


class EncoderBlock(nn.Module):
    """Small transformer block used in the educational vision encoder."""

    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(width)
        self.attn = MultiHeadSelfAttention(width, heads, dropout=dropout)
        self.norm_2 = nn.LayerNorm(width)
        self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm_1(hidden_states))
        hidden_states = hidden_states + self.ffn(self.norm_2(hidden_states))
        return hidden_states


class SoftTokenProjector(nn.Module):
    """Compresses a variable number of source tokens into a fixed token budget."""

    def __init__(self, source_width: int, target_width: int, max_soft_tokens: int, heads: int) -> None:
        super().__init__()
        self.query_bank = nn.Parameter(torch.randn(1, max_soft_tokens, source_width) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=source_width,
            num_heads=heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(source_width)
        self.projection = nn.Sequential(
            nn.Linear(source_width, target_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(target_width, target_width),
        )

    def forward(self, source_tokens: Tensor, num_output_tokens: int) -> Tensor:
        """Returns fixed-length tokens of shape ``[batch, num_output_tokens, target_width]``."""

        queries = self.query_bank[:, :num_output_tokens, :].expand(source_tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, source_tokens, source_tokens, need_weights=False)
        return self.projection(self.norm(attended))


class Gemma4VisionEncoder(nn.Module):
    """Variable-resolution vision tower with 3x3 pooling into soft tokens."""

    def __init__(self, config: Gemma4VisionConfig, llm_width: int) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.row_embedding = nn.Embedding(config.max_axis_positions, config.width)
        self.col_embedding = nn.Embedding(config.max_axis_positions, config.width)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    width=config.width,
                    heads=config.heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.layers)
            ]
        )
        self.soft_token_projector = SoftTokenProjector(
            source_width=config.width,
            target_width=llm_width,
            max_soft_tokens=max(config.allowed_soft_token_budgets),
            heads=config.heads,
        )
        self.image_separator = nn.Parameter(torch.randn(1, 1, llm_width) * 0.02)

    def _resolve_soft_token_budget(self, requested_budget: int | None) -> int:
        """Chooses the nearest supported soft-token budget."""

        if requested_budget is None:
            requested_budget = self.config.soft_tokens_per_image
        return min(
            self.config.allowed_soft_token_budgets,
            key=lambda budget: abs(budget - requested_budget),
        )

    def _choose_patch_grid(
        self,
        image_height: int,
        image_width: int,
        soft_token_budget: int,
    ) -> tuple[int, int]:
        """Chooses a patch grid that respects the Gemma 4 patch-budget idea.

        The production models restrict image sizes so that the patch count fits
        under a configurable budget and the subsequent ``3 x 3`` pooling step
        yields roughly the requested number of soft tokens. This helper mirrors
        that logic by searching patch grids whose dimensions are multiples of the
        pooling kernel.
        """

        patch_budget = soft_token_budget * (self.config.pool_kernel_size**2)
        aspect_ratio = image_width / max(image_height, 1)
        step = self.config.pool_kernel_size

        best_grid = (step, step)
        best_score = float("inf")

        for patch_rows in range(step, patch_budget + 1, step):
            approx_patch_cols = max(
                step,
                int(round((aspect_ratio * patch_rows) / step)) * step,
            )
            for patch_cols in (
                approx_patch_cols - step,
                approx_patch_cols,
                approx_patch_cols + step,
            ):
                if patch_cols < step:
                    continue
                if patch_rows * patch_cols > patch_budget:
                    continue

                coverage = (patch_rows * patch_cols) / patch_budget
                ratio_error = abs((patch_cols / patch_rows) - aspect_ratio)
                score = ratio_error + 0.15 * (1.0 - coverage)
                if score < best_score:
                    best_score = score
                    best_grid = (patch_rows, patch_cols)

        return best_grid

    @staticmethod
    def _resize_with_padding(image: Tensor, target_height: int, target_width: int) -> Tensor:
        """Resizes an image to fit inside a target canvas and pads the remainder."""

        _, image_height, image_width = image.shape
        scale = min(target_height / image_height, target_width / image_width)
        resized_height = max(1, int(round(image_height * scale)))
        resized_width = max(1, int(round(image_width * scale)))
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        canvas = torch.zeros(
            image.shape[0],
            target_height,
            target_width,
            device=image.device,
            dtype=image.dtype,
        )
        top = (target_height - resized_height) // 2
        left = (target_width - resized_width) // 2
        canvas[:, top : top + resized_height, left : left + resized_width] = resized
        return canvas

    def _encode_single_image(self, image: Tensor, soft_token_budget: int) -> tuple[Tensor, tuple[int, int]]:
        """Encodes one image [C, H, W] into a fixed number of soft visual tokens."""

        patch_rows, patch_cols = self._choose_patch_grid(
            image_height=image.shape[1],
            image_width=image.shape[2],
            soft_token_budget=soft_token_budget,
        )
        target_height = patch_rows * self.config.patch_size
        target_width = patch_cols * self.config.patch_size
        resized = self._resize_with_padding(image, target_height, target_width).unsqueeze(0)

        patch_grid = self.patch_embed(resized)
        _, _, grid_height, grid_width = patch_grid.shape
        if grid_height > self.config.max_axis_positions or grid_width > self.config.max_axis_positions:
            raise ValueError(
                "Patch grid exceeds the configured position-embedding capacity: "
                f"({grid_height}, {grid_width}) vs max_axis_positions={self.config.max_axis_positions}."
            )

        row_positions = torch.arange(grid_height, device=image.device)
        col_positions = torch.arange(grid_width, device=image.device)
        row_bias = self.row_embedding(row_positions)[:, None, :]
        col_bias = self.col_embedding(col_positions)[None, :, :]
        position_bias = row_bias + col_bias

        patch_tokens = patch_grid.permute(0, 2, 3, 1).contiguous().view(1, grid_height * grid_width, -1)
        patch_tokens = patch_tokens + position_bias.view(1, grid_height * grid_width, -1)
        for block in self.blocks:
            patch_tokens = block(patch_tokens)

        patch_grid_tokens = patch_tokens.view(1, grid_height, grid_width, -1).permute(0, 3, 1, 2)
        pooled_grid = F.avg_pool2d(
            patch_grid_tokens,
            kernel_size=self.config.pool_kernel_size,
            stride=self.config.pool_kernel_size,
        )
        pooled_tokens = pooled_grid.flatten(2).transpose(1, 2)
        soft_tokens = self.soft_token_projector(pooled_tokens, num_output_tokens=soft_token_budget)
        return soft_tokens.squeeze(0), (patch_rows, patch_cols)

    def encode_images(
        self,
        pixel_values: Tensor | list[Tensor] | list[list[Tensor]],
        *,
        soft_token_budget: int | None = None,
    ) -> tuple[Tensor, Tensor, list[list[tuple[int, int]]], list[int]]:
        """Encodes one or more images into padded visual token sequences."""

        normalized_samples = normalize_image_batch(pixel_values)
        resolved_budget = self._resolve_soft_token_budget(soft_token_budget)

        visual_sequences: list[Tensor] = []
        chosen_patch_grids: list[list[tuple[int, int]]] = []
        num_images_per_sample: list[int] = []

        for sample_images in normalized_samples:
            sample_tokens: list[Tensor] = []
            sample_patch_grids: list[tuple[int, int]] = []

            for image_index, image in enumerate(sample_images):
                if image.ndim != 3:
                    raise ValueError(
                        f"Each image must have shape [channels, height, width], got {tuple(image.shape)}."
                    )
                image_tokens, patch_grid = self._encode_single_image(image, resolved_budget)
                sample_tokens.append(image_tokens)
                sample_patch_grids.append(patch_grid)
                if image_index < len(sample_images) - 1:
                    sample_tokens.append(self.image_separator.squeeze(0))

            sample_sequence = torch.cat(sample_tokens, dim=0)
            visual_sequences.append(sample_sequence)
            chosen_patch_grids.append(sample_patch_grids)
            num_images_per_sample.append(len(sample_images))

        max_visual_len = max(sequence.shape[0] for sequence in visual_sequences)
        hidden_width = visual_sequences[0].shape[-1]
        padded_visual_tokens = torch.zeros(
            len(visual_sequences),
            max_visual_len,
            hidden_width,
            device=visual_sequences[0].device,
            dtype=visual_sequences[0].dtype,
        )
        visual_attention_mask = torch.zeros(
            len(visual_sequences),
            max_visual_len,
            device=visual_sequences[0].device,
            dtype=torch.long,
        )

        for index, sequence in enumerate(visual_sequences):
            length = sequence.shape[0]
            padded_visual_tokens[index, :length] = sequence
            visual_attention_mask[index, :length] = 1

        return padded_visual_tokens, visual_attention_mask, chosen_patch_grids, num_images_per_sample


class DepthwiseConvModule(nn.Module):
    """A small conformer-like depthwise convolution block for audio."""

    def __init__(self, width: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.pointwise_in = nn.Conv1d(width, width * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=padding,
            groups=width,
        )
        self.pointwise_out = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.glu(self.pointwise_in(hidden_states), dim=1)
        hidden_states = self.depthwise(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.pointwise_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states.transpose(1, 2)


class AudioEncoderBlock(nn.Module):
    """Conformer-like block used in the educational audio encoder."""

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        conv_kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(width)
        self.attn = MultiHeadSelfAttention(width, heads, dropout=dropout)
        self.norm_2 = nn.LayerNorm(width)
        self.conv = DepthwiseConvModule(width, conv_kernel_size, dropout=dropout)
        self.norm_3 = nn.LayerNorm(width)
        self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm_1(hidden_states))
        hidden_states = hidden_states + self.conv(self.norm_2(hidden_states))
        hidden_states = hidden_states + self.ffn(self.norm_3(hidden_states))
        return hidden_states


class Gemma4AudioEncoder(nn.Module):
    """USM-style audio tower approximation for Gemma 4 E2B / E4B."""

    def __init__(self, config: Gemma4AudioConfig, llm_width: int) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.width)
        self.subsample = nn.Conv1d(config.width, config.width, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                AudioEncoderBlock(
                    width=config.width,
                    heads=config.heads,
                    mlp_ratio=config.mlp_ratio,
                    conv_kernel_size=config.conv_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.layers)
            ]
        )
        self.token_projector = SoftTokenProjector(
            source_width=config.width,
            target_width=llm_width,
            max_soft_tokens=config.compress_tokens_per_clip,
            heads=config.heads,
        )
        self.audio_separator = nn.Parameter(torch.randn(1, 1, llm_width) * 0.02)

    def _encode_single_clip(self, clip: Tensor) -> Tensor:
        """Encodes one audio clip into a fixed number of audio tokens."""

        if clip.ndim != 2:
            raise ValueError(f"Audio clip must have shape [time, feature_dim], got {tuple(clip.shape)}.")

        hidden_states = self.input_projection(clip).unsqueeze(0)
        hidden_states = self.subsample(hidden_states.transpose(1, 2)).transpose(1, 2)
        positions = build_sinusoidal_positions(
            sequence_length=hidden_states.shape[1],
            width=hidden_states.shape[2],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_states = hidden_states + positions.unsqueeze(0)

        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.token_projector(
            hidden_states,
            num_output_tokens=self.config.compress_tokens_per_clip,
        ).squeeze(0)

    def encode_audio(self, audio_values: Tensor | list[Tensor]) -> tuple[Tensor, Tensor, list[int]]:
        """Encodes one or more audio clips into padded token sequences."""

        normalized_samples = normalize_audio_batch(audio_values)
        audio_sequences: list[Tensor] = []
        audio_lengths: list[int] = []

        for sample_clips in normalized_samples:
            sample_tokens: list[Tensor] = []
            for clip_index, clip in enumerate(sample_clips):
                clip_tokens = self._encode_single_clip(clip)
                sample_tokens.append(clip_tokens)
                if clip_index < len(sample_clips) - 1:
                    sample_tokens.append(self.audio_separator.squeeze(0))

            sample_sequence = torch.cat(sample_tokens, dim=0)
            audio_sequences.append(sample_sequence)
            audio_lengths.append(sample_sequence.shape[0])

        max_audio_len = max(sequence.shape[0] for sequence in audio_sequences)
        hidden_width = audio_sequences[0].shape[-1]
        padded_audio_tokens = torch.zeros(
            len(audio_sequences),
            max_audio_len,
            hidden_width,
            device=audio_sequences[0].device,
            dtype=audio_sequences[0].dtype,
        )
        audio_attention_mask = torch.zeros(
            len(audio_sequences),
            max_audio_len,
            device=audio_sequences[0].device,
            dtype=torch.long,
        )

        for index, sequence in enumerate(audio_sequences):
            length = sequence.shape[0]
            padded_audio_tokens[index, :length] = sequence
            audio_attention_mask[index, :length] = 1

        return padded_audio_tokens, audio_attention_mask, audio_lengths


class Gemma4PerLayerEmbedding(nn.Module):
    """Compact educational approximation of Gemma 4 Per-Layer Embeddings.

    Production Gemma 4 uses a per-layer side-channel that depends on token IDs
    and context. To avoid creating a huge per-layer vocabulary table, this
    educational implementation hashes token IDs into a smaller shared side table
    and combines that with per-layer projections of the initial token embeddings.
    """

    def __init__(
        self,
        *,
        vocab_hash_buckets: int,
        side_width: int,
        hidden_width: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.vocab_hash_buckets = vocab_hash_buckets
        self.token_id_table = nn.Embedding(vocab_hash_buckets, side_width)
        self.token_scales = nn.Parameter(torch.ones(num_layers, side_width))
        self.context_projections = nn.ModuleList(
            [nn.Linear(hidden_width, side_width, bias=False) for _ in range(num_layers)]
        )
        self.context_norm = nn.ModuleList([RMSNorm(side_width) for _ in range(num_layers)])
        self.output_projections = nn.ModuleList(
            [nn.Linear(side_width, hidden_width, bias=False) for _ in range(num_layers)]
        )

    def forward(self, input_ids: Tensor, base_text_embeddings: Tensor, layer_index: int) -> Tensor:
        """Returns the layer-specific additive side input for one decoder layer."""

        hashed_ids = input_ids.remainder(self.vocab_hash_buckets)
        token_id_signal = self.token_id_table(hashed_ids) * self.token_scales[layer_index]
        context_signal = self.context_norm[layer_index](self.context_projections[layer_index](base_text_embeddings))
        side_input = (token_id_signal + context_signal) / math.sqrt(2.0)
        return self.output_projections[layer_index](side_input)


class Gemma4GQAAttention(nn.Module):
    """Grouped-query attention with Gemma 4 style local/global layer modes."""

    def __init__(
        self,
        *,
        width: int,
        num_query_heads: int,
        num_kv_heads: int,
        local_head_dim: int,
        global_head_dim: int,
        is_global: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")

        self.width = width
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.is_global = is_global
        self.head_dim = global_head_dim if is_global else local_head_dim
        self.kv_repeat = num_query_heads // num_kv_heads
        self.scale = self.head_dim**-0.5

        self.query_width = num_query_heads * self.head_dim
        self.kv_width = num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(width, self.query_width, bias=False)
        self.k_proj = nn.Linear(width, self.kv_width, bias=False)
        self.v_proj = nn.Linear(width, self.kv_width, bias=False)
        self.out_proj = nn.Linear(self.query_width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _build_structural_mask(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        local_window_size: int | None,
    ) -> Tensor:
        """Builds a causal mask with optional sliding-window restriction."""

        key_positions = torch.arange(sequence_length, device=device)
        query_positions = torch.arange(sequence_length, device=device).unsqueeze(-1)
        causal_mask = key_positions > query_positions

        if local_window_size is not None:
            too_old_mask = key_positions < (query_positions - local_window_size + 1)
            causal_mask = causal_mask | too_old_mask

        return causal_mask

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        local_window_size: int | None = None,
        rope_base: float = 10_000.0,
    ) -> Tensor:
        """Runs GQA in local or global mode."""

        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_query_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = build_rope_cache(
            sequence_length=sequence_length,
            head_dim=self.head_dim,
            base=rope_base,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        key = key.repeat_interleave(self.kv_repeat, dim=1)
        value = value.repeat_interleave(self.kv_repeat, dim=1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        structural_mask = self._build_structural_mask(
            sequence_length=sequence_length,
            device=hidden_states.device,
            local_window_size=local_window_size,
        )
        min_value = torch.finfo(attention_scores.dtype).min
        attention_scores = attention_scores.masked_fill(
            structural_mask.unsqueeze(0).unsqueeze(0),
            min_value,
        )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape [batch, seq_len], "
                    f"got {tuple(attention_mask.shape)}."
                )
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            attention_scores = attention_scores.masked_fill(~key_mask, min_value)

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.query_width)
        return self.out_proj(context)


class Gemma4DecoderBlock(nn.Module):
    """Decoder block with local/global attention selected by ``layer_type``."""

    def __init__(self, config: Gemma4TextConfig, *, layer_type: str) -> None:
        super().__init__()
        if layer_type not in {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        self.layer_type = layer_type
        self.local_window_size = (
            None if layer_type == "full_attention" else config.local_window_size
        )
        self.rope_base = (
            config.global_rope_base if layer_type == "full_attention" else config.local_rope_base
        )
        self.norm_1 = RMSNorm(config.width)
        self.attn = Gemma4GQAAttention(
            width=config.width,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            local_head_dim=config.head_dim,
            global_head_dim=config.global_head_dim,
            is_global=(layer_type == "full_attention"),
            dropout=config.dropout,
        )
        self.norm_2 = RMSNorm(config.width)
        self.ffn = FeedForward(config.width, config.mlp_ratio, dropout=config.dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm_1(hidden_states),
            attention_mask=attention_mask,
            local_window_size=self.local_window_size,
            rope_base=self.rope_base,
        )
        hidden_states = hidden_states + self.ffn(self.norm_2(hidden_states))
        return hidden_states


class Gemma4LanguageModel(nn.Module):
    """Decoder-only Gemma 4 backbone with hybrid attention and PLE."""

    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        if len(config.layer_types) != config.layers:
            raise ValueError(
                f"Expected {config.layers} layer types, got {len(config.layer_types)}."
            )

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.ple = Gemma4PerLayerEmbedding(
            vocab_hash_buckets=config.ple_hash_buckets,
            side_width=config.hidden_size_per_layer_input,
            hidden_width=config.width,
            num_layers=config.layers,
        )
        self.blocks = nn.ModuleList(
            [Gemma4DecoderBlock(config, layer_type=layer_type) for layer_type in config.layer_types]
        )
        self.final_norm = RMSNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(
        self,
        *,
        input_ids: Tensor,
        multimodal_tokens: Tensor,
        multimodal_attention_mask: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Runs causal decoding over multimodal-prefix and text tokens."""

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, text_len], got {tuple(input_ids.shape)}.")
        if multimodal_tokens.ndim != 3:
            raise ValueError(
                "multimodal_tokens must have shape [batch, prefix_len, hidden], "
                f"got {tuple(multimodal_tokens.shape)}."
            )
        if multimodal_attention_mask.ndim != 2:
            raise ValueError(
                "multimodal_attention_mask must have shape [batch, prefix_len], "
                f"got {tuple(multimodal_attention_mask.shape)}."
            )

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        text_embeddings = self.token_embedding(input_ids)
        hidden_states = torch.cat([multimodal_tokens, text_embeddings], dim=1)
        full_attention_mask = torch.cat([multimodal_attention_mask, attention_mask], dim=1)
        prefix_length = multimodal_tokens.shape[1]

        for layer_index, block in enumerate(self.blocks):
            ple_delta = self.ple(input_ids, text_embeddings, layer_index)
            if prefix_length > 0:
                prefix_pad = torch.zeros(
                    input_ids.shape[0],
                    prefix_length,
                    self.config.width,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                full_ple_delta = torch.cat([prefix_pad, ple_delta], dim=1)
            else:
                full_ple_delta = ple_delta

            hidden_states = hidden_states + full_ple_delta
            hidden_states = block(hidden_states, attention_mask=full_attention_mask)

        hidden_states = self.final_norm(hidden_states)
        text_hidden_states = hidden_states[:, prefix_length:, :]
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


class Gemma4Model(nn.Module):
    """Educational multimodal Gemma 4 model covering the E2B / E4B path."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = Gemma4VisionEncoder(config.vision_config, llm_width=config.text_config.width)
        self.audio_encoder = (
            Gemma4AudioEncoder(config.audio_config, llm_width=config.text_config.width)
            if config.supports_audio
            else None
        )
        self.language_model = Gemma4LanguageModel(config.text_config)
        self.image_type_embedding = nn.Parameter(torch.randn(1, 1, config.text_config.width) * 0.02)
        self.audio_type_embedding = nn.Parameter(torch.randn(1, 1, config.text_config.width) * 0.02)

    def encode_images(
        self,
        pixel_values: Tensor | list[Tensor] | list[list[Tensor]],
        *,
        soft_token_budget: int | None = None,
    ) -> tuple[Tensor, Tensor, list[list[tuple[int, int]]], list[int]]:
        """Encodes images into padded visual token sequences."""

        return self.vision_encoder.encode_images(pixel_values, soft_token_budget=soft_token_budget)

    def encode_audio(
        self,
        audio_values: Tensor | list[Tensor],
    ) -> tuple[Tensor, Tensor, list[int]]:
        """Encodes audio clips into padded token sequences."""

        if self.audio_encoder is None:
            raise ValueError("This Gemma 4 configuration does not include the audio tower.")
        return self.audio_encoder.encode_audio(audio_values)

    def _empty_prefix(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Returns an empty prefix tensor and mask for text-only decoding."""

        tokens = torch.zeros(batch_size, 0, self.config.text_config.width, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, 0, device=device, dtype=torch.long)
        return tokens, mask

    def forward(
        self,
        *,
        input_ids: Tensor,
        pixel_values: Tensor | list[Tensor] | list[list[Tensor]] | None = None,
        audio_values: Tensor | list[Tensor] | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        soft_token_budget: int | None = None,
    ) -> Gemma4Output:
        """Runs multimodal Gemma 4 decoding.

        The model treats visual and audio tokens as a prefix placed before the
        text sequence. This mirrors the common decoder-only multimodal pattern
        used by many open VLMs while keeping the implementation easy to inspect.
        """

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, text_len], got {tuple(input_ids.shape)}.")

        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.language_model.token_embedding.weight.dtype

        chosen_patch_grids: list[list[tuple[int, int]]] = [[] for _ in range(batch_size)]
        num_images_per_sample = [0 for _ in range(batch_size)]
        audio_token_lengths = [0 for _ in range(batch_size)]

        if pixel_values is None:
            visual_tokens, visual_attention_mask = self._empty_prefix(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        else:
            visual_tokens, visual_attention_mask, chosen_patch_grids, num_images_per_sample = self.encode_images(
                pixel_values,
                soft_token_budget=soft_token_budget,
            )
            visual_tokens = visual_tokens + self.image_type_embedding

        if audio_values is None:
            audio_tokens, audio_attention_mask = self._empty_prefix(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        else:
            if self.audio_encoder is None:
                raise ValueError("audio_values were provided but this model configuration has no audio tower.")
            audio_tokens, audio_attention_mask, audio_token_lengths = self.encode_audio(audio_values)
            audio_tokens = audio_tokens + self.audio_type_embedding

        multimodal_tokens = torch.cat([visual_tokens, audio_tokens], dim=1)
        multimodal_attention_mask = torch.cat([visual_attention_mask, audio_attention_mask], dim=1)
        logits, loss = self.language_model(
            input_ids=input_ids,
            multimodal_tokens=multimodal_tokens,
            multimodal_attention_mask=multimodal_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
        )

        return Gemma4Output(
            visual_tokens=visual_tokens,
            visual_attention_mask=visual_attention_mask,
            audio_tokens=audio_tokens,
            audio_attention_mask=audio_attention_mask,
            logits=logits,
            loss=loss,
            chosen_patch_grids=chosen_patch_grids,
            num_images_per_sample=num_images_per_sample,
            audio_token_lengths=audio_token_lengths,
        )


def make_gemma_4_e2b_reference_config() -> Gemma4Config:
    """Returns a reference-style config mirroring official E2B dimensions.

    This config is provided for study and comparison with the official model
    card. It is not meant for lightweight local smoke tests.
    """

    return Gemma4Config(
        variant_name="Gemma4-E2B-reference",
        supports_audio=True,
        vision_config=Gemma4VisionConfig(
            patch_size=16,
            pool_kernel_size=3,
            soft_tokens_per_image=280,
            allowed_soft_token_budgets=(70, 140, 280, 560, 1120),
            width=768,
            layers=16,
            heads=12,
            mlp_ratio=4.0,
            dropout=0.0,
            max_axis_positions=512,
        ),
        audio_config=Gemma4AudioConfig(
            feature_dim=80,
            width=1024,
            layers=12,
            heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            conv_kernel_size=5,
            compress_tokens_per_clip=64,
            max_positions=4096,
        ),
        text_config=Gemma4TextConfig(
            vocab_size=262_144,
            context_length=128_000,
            width=1536,
            layers=35,
            num_query_heads=8,
            num_kv_heads=1,
            head_dim=256,
            global_head_dim=512,
            local_window_size=512,
            layer_types=repeat_attention_pattern(4, 7),
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
            local_rope_base=10_000.0,
            global_rope_base=1_000_000.0,
            hidden_size_per_layer_input=256,
            ple_hash_buckets=8192,
            num_kv_shared_layers=20,
        ),
    )


def make_gemma_4_e4b_reference_config() -> Gemma4Config:
    """Returns a reference-style config mirroring official E4B dimensions."""

    return Gemma4Config(
        variant_name="Gemma4-E4B-reference",
        supports_audio=True,
        vision_config=Gemma4VisionConfig(
            patch_size=16,
            pool_kernel_size=3,
            soft_tokens_per_image=280,
            allowed_soft_token_budgets=(70, 140, 280, 560, 1120),
            width=768,
            layers=16,
            heads=12,
            mlp_ratio=4.0,
            dropout=0.0,
            max_axis_positions=512,
        ),
        audio_config=Gemma4AudioConfig(
            feature_dim=80,
            width=1024,
            layers=12,
            heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            conv_kernel_size=5,
            compress_tokens_per_clip=64,
            max_positions=4096,
        ),
        text_config=Gemma4TextConfig(
            vocab_size=262_144,
            context_length=128_000,
            width=2560,
            layers=42,
            num_query_heads=8,
            num_kv_heads=2,
            head_dim=256,
            global_head_dim=512,
            local_window_size=512,
            layer_types=repeat_attention_pattern(5, 7),
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
            local_rope_base=10_000.0,
            global_rope_base=1_000_000.0,
            hidden_size_per_layer_input=256,
            ple_hash_buckets=8192,
            num_kv_shared_layers=18,
        ),
    )


def build_gemma_4_e2b_tiny() -> Gemma4Model:
    """Builds a small E2B-style model for smoke tests and study."""

    config = Gemma4Config(
        variant_name="Gemma4-E2B-tiny",
        supports_audio=True,
        vision_config=Gemma4VisionConfig(
            patch_size=8,
            pool_kernel_size=3,
            soft_tokens_per_image=16,
            allowed_soft_token_budgets=(16, 32, 64),
            width=96,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            max_axis_positions=96,
        ),
        audio_config=Gemma4AudioConfig(
            feature_dim=40,
            width=96,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            conv_kernel_size=5,
            compress_tokens_per_clip=12,
            max_positions=256,
        ),
        text_config=Gemma4TextConfig(
            vocab_size=512,
            context_length=64,
            width=192,
            layers=10,
            num_query_heads=4,
            num_kv_heads=1,
            head_dim=48,
            global_head_dim=64,
            local_window_size=8,
            layer_types=repeat_attention_pattern(4, 2),
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
            local_rope_base=10_000.0,
            global_rope_base=1_000_000.0,
            hidden_size_per_layer_input=48,
            ple_hash_buckets=512,
            num_kv_shared_layers=4,
        ),
    )
    return Gemma4Model(config)


def build_gemma_4_e4b_tiny() -> Gemma4Model:
    """Builds a small E4B-style model for smoke tests and study."""

    config = Gemma4Config(
        variant_name="Gemma4-E4B-tiny",
        supports_audio=True,
        vision_config=Gemma4VisionConfig(
            patch_size=8,
            pool_kernel_size=3,
            soft_tokens_per_image=16,
            allowed_soft_token_budgets=(16, 32, 64),
            width=112,
            layers=3,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            max_axis_positions=96,
        ),
        audio_config=Gemma4AudioConfig(
            feature_dim=40,
            width=112,
            layers=3,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            conv_kernel_size=5,
            compress_tokens_per_clip=12,
            max_positions=256,
        ),
        text_config=Gemma4TextConfig(
            vocab_size=512,
            context_length=64,
            width=256,
            layers=12,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=48,
            global_head_dim=64,
            local_window_size=8,
            layer_types=repeat_attention_pattern(5, 2),
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
            local_rope_base=10_000.0,
            global_rope_base=1_000_000.0,
            hidden_size_per_layer_input=64,
            ple_hash_buckets=512,
            num_kv_shared_layers=4,
        ),
    )
    return Gemma4Model(config)


def _run_smoke_test(model: Gemma4Model) -> None:
    """Runs one small forward pass and prints key tensor shapes."""

    images = [
        [torch.randn(3, 48, 72), torch.randn(3, 72, 48)],
        [torch.randn(3, 64, 64)],
    ]
    audio = [
        torch.randn(40, model.config.audio_config.feature_dim),
        torch.randn(52, model.config.audio_config.feature_dim),
    ]
    input_ids = torch.tensor(
        [
            [11, 21, 31, 41, 51, 61, 0, 0],
            [12, 22, 32, 42, 52, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()
    labels = input_ids.masked_fill(input_ids == 0, -100)

    output = model(
        input_ids=input_ids,
        pixel_values=images,
        audio_values=audio,
        attention_mask=attention_mask,
        labels=labels,
    )
    print("variant:", model.config.variant_name)
    print("visual_tokens:", tuple(output.visual_tokens.shape))
    print("visual_attention_mask:", tuple(output.visual_attention_mask.shape))
    print("audio_tokens:", tuple(output.audio_tokens.shape))
    print("audio_attention_mask:", tuple(output.audio_attention_mask.shape))
    print("logits:", tuple(output.logits.shape))
    print("chosen_patch_grids:", output.chosen_patch_grids)
    print("num_images_per_sample:", output.num_images_per_sample)
    print("audio_token_lengths:", output.audio_token_lengths)
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _run_smoke_test(build_gemma_4_e2b_tiny())
    print("=" * 80)
    _run_smoke_test(build_gemma_4_e4b_tiny())
