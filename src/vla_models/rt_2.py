"""A compact educational RT-2 implementation in PyTorch.

This module follows the architecture summary in ``docs/VLAs/RT_2.md``:

1. A pretrained-style vision-language backbone receives images and instruction
   text in one shared token space.
2. Low-level robot actions are represented as text-like tokens rather than a
   separate continuous action head.
3. The policy is trained with the same autoregressive next-token objective that
   is used by ordinary language models.
4. Robot data and web-style language targets can be optimized together through a
   co-fine-tuning interface.

This implementation is intentionally educational rather than checkpoint-faithful:

* The vision encoder is a compact ViT-style patch encoder, not ViT-22B or ViT-4B.
* The PaLI-X and PaLM-E families are approximated with two backbone modes:
  encoder-decoder and decoder-only.
* Action tokens are modeled with one reserved vocabulary range instead of exact
  tokenizer surgery on production VLM vocabularies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RT2VisionConfig:
    """Configuration for the lightweight visual tokenizer.

    Attributes:
        image_size: Square resolution expected by the educational encoder.
        in_channels: Number of input image channels.
        patch_size: Patch size used by the ViT-style stem.
        width: Hidden width inside the visual encoder.
        layers: Number of self-attention blocks in the visual encoder.
        heads: Number of attention heads inside the visual encoder.
        mlp_ratio: Expansion ratio of the visual MLP block.
        dropout: Dropout used in the visual encoder.
        max_images: Maximum number of images supported in one prompt.
    """

    image_size: int = 256
    in_channels: int = 3
    patch_size: int = 16
    width: int = 256
    layers: int = 4
    heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_images: int = 1

    @property
    def patches_per_image(self) -> int:
        """Returns the number of patch tokens emitted per image."""

        grid_size = self.image_size // self.patch_size
        return grid_size * grid_size

    @property
    def max_visual_tokens(self) -> int:
        """Returns the maximum total visual token count."""

        return self.max_images * self.patches_per_image


@dataclass(frozen=True)
class RT2TextConfig:
    """Configuration for text prompts and autoregressive decoding.

    Attributes:
        vocab_size: Total vocabulary size used by the model.
        pad_token_id: Padding token ID.
        bos_token_id: BOS token ID used for teacher forcing and generation.
        eos_token_id: EOS token ID used for optional natural language targets.
        max_prompt_tokens: Maximum prompt length fed to the model.
        max_target_tokens: Maximum decoded target length.
    """

    vocab_size: int = 32_000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_prompt_tokens: int = 64
    max_target_tokens: int = 64


@dataclass(frozen=True)
class RT2ActionConfig:
    """Configuration for RT-2-style action tokenization.

    The paper discretizes all continuous action dimensions into 256 bins and
    writes them out as tokens. This educational implementation reserves one
    contiguous action-token range and uses it for:

    * token 0 / 1 inside the range for the termination flag
    * tokens 0..255 inside the range for each continuous action dimension

    Attributes:
        action_token_offset: The first token ID reserved for action bins.
        action_bins: Number of available action bins.
        continuous_min: Lower bound used when binning continuous actions.
        continuous_max: Upper bound used when binning continuous actions.
    """

    action_token_offset: int = 31_744
    action_bins: int = 256
    continuous_min: float = -1.0
    continuous_max: float = 1.0

    @property
    def continuous_names(self) -> tuple[str, ...]:
        """Returns the ordered continuous action dimensions."""

        return (
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_roll",
            "delta_pitch",
            "delta_yaw",
            "gripper",
        )

    @property
    def action_names(self) -> tuple[str, ...]:
        """Returns the ordered names of all decoded action tokens."""

        return ("terminate",) + self.continuous_names

    @property
    def action_sequence_length(self) -> int:
        """Returns the number of action tokens emitted per step."""

        return len(self.action_names)


@dataclass(frozen=True)
class RT2TransformerConfig:
    """Configuration for the language backbone.

    Attributes:
        width: Hidden width of the multimodal transformer.
        encoder_layers: Number of encoder layers used by the encoder-decoder
            approximation of RT-2-PaLI-X.
        decoder_layers: Number of decoder layers used by both backbone modes.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio inside feed-forward blocks.
        dropout: Dropout used in attention and MLP layers.
        backbone_type: Either ``"encoder_decoder"`` or ``"decoder_only"``.
    """

    width: int = 512
    encoder_layers: int = 4
    decoder_layers: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    backbone_type: str = "encoder_decoder"


@dataclass(frozen=True)
class RT2Config:
    """Top-level configuration for the educational RT-2 model."""

    vision_config: RT2VisionConfig = field(default_factory=RT2VisionConfig)
    text_config: RT2TextConfig = field(default_factory=RT2TextConfig)
    action_config: RT2ActionConfig = field(default_factory=RT2ActionConfig)
    transformer_config: RT2TransformerConfig = field(default_factory=RT2TransformerConfig)


@dataclass
class RT2Output:
    """Container returned by ``RT2Model.forward``.

    Attributes:
        visual_tokens: Projected vision tokens with shape ``[batch, V, width]``.
        prompt_embeddings: Prompt token embeddings with shape ``[batch, T, width]``.
        backbone_inputs: Embeddings fed into the main backbone. For the
            encoder-decoder mode this is the context sequence. For the
            decoder-only mode this is the full fused input sequence.
        decoder_input_ids: Teacher-forced input IDs used to predict the target.
        logits: Token logits aligned with the target sequence.
        loss: Optional masked next-token loss.
    """

    visual_tokens: Tensor
    prompt_embeddings: Tensor
    backbone_inputs: Tensor
    decoder_input_ids: Tensor | None
    logits: Tensor | None
    loss: Tensor | None = None


@dataclass
class RT2CoFineTuneOutput:
    """Container returned by ``RT2Model.co_finetune_step``.

    Attributes:
        robot_output: Forward output for the robot-action batch.
        web_output: Forward output for the web-style language batch.
        loss: Weighted sum of robot and web losses.
    """

    robot_output: RT2Output
    web_output: RT2Output
    loss: Tensor


class RMSNorm(nn.Module):
    """Root-mean-square normalization used across all transformer blocks."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class MultiHeadAttention(nn.Module):
    """Minimal multi-head attention supporting self- and cross-attention."""

    def __init__(self, width: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")

        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, hidden_states: Tensor) -> Tensor:
        """Reshapes a sequence to ``[batch, heads, seq, head_dim]``."""

        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return hidden_states.transpose(1, 2)

    def forward(
        self,
        query_states: Tensor,
        *,
        key_value_states: Tensor | None = None,
        query_mask: Tensor | None = None,
        key_value_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Runs attention over the provided states."""

        if key_value_states is None:
            key_value_states = query_states
        if key_value_mask is None:
            key_value_mask = query_mask

        query = self._reshape(self.q_proj(query_states))
        key = self._reshape(self.k_proj(key_value_states))
        value = self._reshape(self.v_proj(key_value_states))

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if key_value_mask is not None:
            key_padding_mask = ~key_value_mask[:, None, None, :]
            attention_scores = attention_scores.masked_fill(
                key_padding_mask,
                torch.finfo(attention_scores.dtype).min,
            )

        if causal:
            query_length = query_states.size(1)
            key_length = key_value_states.size(1)
            causal_mask = torch.triu(
                torch.ones(query_length, key_length, device=query_states.device, dtype=torch.bool),
                diagonal=1 + max(0, key_length - query_length),
            )
            attention_scores = attention_scores.masked_fill(
                causal_mask[None, None, :, :],
                torch.finfo(attention_scores.dtype).min,
            )

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(
            query_states.size(0),
            query_states.size(1),
            self.width,
        )
        context = self.out_proj(context)

        if query_mask is not None:
            context = context * query_mask.unsqueeze(-1).to(context.dtype)
        return context


class FeedForward(nn.Module):
    """Transformer feed-forward block with GELU."""

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


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm encoder block used in the visual and context encoders."""

    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(width)
        self.self_attn = MultiHeadAttention(width, heads, dropout=dropout)
        self.norm_2 = RMSNorm(width)
        self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)

    def forward(self, hidden_states: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.norm_1(hidden_states),
            query_mask=attention_mask,
            key_value_mask=attention_mask,
            causal=False,
        )
        hidden_states = hidden_states + self.ffn(self.norm_2(hidden_states))
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class TransformerDecoderBlock(nn.Module):
    """Decoder block with causal self-attention and optional cross-attention."""

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        *,
        dropout: float = 0.0,
        use_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(width)
        self.self_attn = MultiHeadAttention(width, heads, dropout=dropout)
        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.norm_2 = RMSNorm(width)
            self.cross_attn = MultiHeadAttention(width, heads, dropout=dropout)
            self.norm_3 = RMSNorm(width)
            self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)
        else:
            self.norm_2 = RMSNorm(width)
            self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Runs one decoder block."""

        hidden_states = hidden_states + self.self_attn(
            self.norm_1(hidden_states),
            query_mask=attention_mask,
            key_value_mask=attention_mask,
            causal=True,
        )

        if self.use_cross_attention:
            if memory is None:
                raise ValueError("Cross-attention blocks require memory states.")
            hidden_states = hidden_states + self.cross_attn(
                self.norm_2(hidden_states),
                key_value_states=memory,
                query_mask=attention_mask,
                key_value_mask=memory_mask,
                causal=False,
            )
            hidden_states = hidden_states + self.ffn(self.norm_3(hidden_states))
        else:
            hidden_states = hidden_states + self.ffn(self.norm_2(hidden_states))

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states


class VisionPatchEmbed(nn.Module):
    """Projects images to patch tokens with a stride-equals-kernel stem."""

    def __init__(self, config: RT2VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            config.in_channels,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Returns patch tokens with shape ``[batch, num_patches, width]``."""

        hidden_states = self.proj(pixel_values)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class RT2VisionEncoder(nn.Module):
    """Small ViT-like encoder that approximates the VLM image tower."""

    def __init__(self, config: RT2VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, config.patches_per_image, config.width)
        )
        self.frame_embed = nn.Parameter(torch.zeros(1, config.max_images, 1, config.width))
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.width,
                    config.heads,
                    config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.layers)
            ]
        )
        self.norm = RMSNorm(config.width)

        nn.init.normal_(self.patch_pos_embed, std=0.02)
        nn.init.normal_(self.frame_embed, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        """Encodes one or more images into patch tokens.

        Args:
            images: Tensor with shape ``[batch, num_images, C, H, W]``.

        Returns:
            Tensor with shape ``[batch, num_images * patches_per_image, width]``.
        """

        batch_size, num_images, channels, height, width = images.shape
        if num_images > self.config.max_images:
            raise ValueError(
                f"num_images={num_images} exceeds max_images={self.config.max_images}."
            )
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                "This educational encoder expects already-resized square images "
                f"of shape {self.config.image_size}x{self.config.image_size}."
            )

        flat_images = images.view(batch_size * num_images, channels, height, width)
        hidden_states = self.patch_embed(flat_images)
        hidden_states = hidden_states + self.patch_pos_embed

        patch_count = hidden_states.size(1)
        hidden_states = hidden_states.reshape(batch_size, num_images, patch_count, self.config.width)
        hidden_states = hidden_states + self.frame_embed[:, :num_images]
        hidden_states = hidden_states.reshape(batch_size * num_images, patch_count, self.config.width)

        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, num_images * patch_count, self.config.width)
        return hidden_states


class RT2EncoderDecoderBackbone(nn.Module):
    """Approximates the PaLI-X style encoder-decoder VLM."""

    def __init__(self, config: RT2Config) -> None:
        super().__init__()
        width = config.transformer_config.width
        self.max_context_tokens = (
            config.vision_config.max_visual_tokens + config.text_config.max_prompt_tokens
        )
        self.max_target_tokens = config.text_config.max_target_tokens

        self.context_pos_embed = nn.Parameter(torch.zeros(1, self.max_context_tokens, width))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.max_target_tokens, width))

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    width,
                    config.transformer_config.heads,
                    config.transformer_config.mlp_ratio,
                    dropout=config.transformer_config.dropout,
                )
                for _ in range(config.transformer_config.encoder_layers)
            ]
        )
        self.encoder_norm = RMSNorm(width)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    width,
                    config.transformer_config.heads,
                    config.transformer_config.mlp_ratio,
                    dropout=config.transformer_config.dropout,
                    use_cross_attention=True,
                )
                for _ in range(config.transformer_config.decoder_layers)
            ]
        )
        self.decoder_norm = RMSNorm(width)

        nn.init.normal_(self.context_pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def forward(
        self,
        context_embeddings: Tensor,
        *,
        context_mask: Tensor,
        decoder_embeddings: Tensor,
        decoder_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs encoder-decoder inference.

        Returns:
            A tuple ``(encoded_context, decoded_states)``.
        """

        context_length = context_embeddings.size(1)
        target_length = decoder_embeddings.size(1)
        if context_length > self.max_context_tokens:
            raise ValueError(
                f"context_length={context_length} exceeds max={self.max_context_tokens}."
            )
        if target_length > self.max_target_tokens:
            raise ValueError(
                f"target_length={target_length} exceeds max={self.max_target_tokens}."
            )

        context_embeddings = context_embeddings + self.context_pos_embed[:, :context_length]
        hidden_states = context_embeddings
        for block in self.encoder_blocks:
            hidden_states = block(hidden_states, attention_mask=context_mask)
        memory = self.encoder_norm(hidden_states)

        decoder_embeddings = decoder_embeddings + self.decoder_pos_embed[:, :target_length]
        hidden_states = decoder_embeddings
        for block in self.decoder_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=decoder_mask,
                memory=memory,
                memory_mask=context_mask,
            )
        hidden_states = self.decoder_norm(hidden_states)
        return memory, hidden_states


class RT2DecoderOnlyBackbone(nn.Module):
    """Approximates the PaLM-E style decoder-only VLM."""

    def __init__(self, config: RT2Config) -> None:
        super().__init__()
        width = config.transformer_config.width
        self.max_seq_tokens = (
            config.vision_config.max_visual_tokens
            + config.text_config.max_prompt_tokens
            + config.text_config.max_target_tokens
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_tokens, width))
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    width,
                    config.transformer_config.heads,
                    config.transformer_config.mlp_ratio,
                    dropout=config.transformer_config.dropout,
                    use_cross_attention=False,
                )
                for _ in range(config.transformer_config.decoder_layers)
            ]
        )
        self.norm = RMSNorm(width)

        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, sequence_embeddings: Tensor, *, sequence_mask: Tensor) -> Tensor:
        """Runs decoder-only inference over the fused multimodal sequence."""

        sequence_length = sequence_embeddings.size(1)
        if sequence_length > self.max_seq_tokens:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds max={self.max_seq_tokens}."
            )

        hidden_states = sequence_embeddings + self.pos_embed[:, :sequence_length]
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=sequence_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class RT2ActionTokenizer:
    """Converts continuous robot actions to RT-2-style token IDs."""

    def __init__(self, config: RT2ActionConfig) -> None:
        self.config = config

    def encode(self, continuous_actions: Tensor, terminate: Tensor) -> Tensor:
        """Encodes actions to token IDs.

        Args:
            continuous_actions: Tensor with shape ``[batch, 7]`` normalized to
                the configured continuous range.
            terminate: Tensor with shape ``[batch]`` containing 0/1 flags.

        Returns:
            Tensor with shape ``[batch, 8]`` containing action token IDs.
        """

        expected_dims = len(self.config.continuous_names)
        if continuous_actions.ndim != 2 or continuous_actions.size(-1) != expected_dims:
            raise ValueError(
                "continuous_actions must have shape "
                f"[batch, {expected_dims}], got {tuple(continuous_actions.shape)}."
            )

        low = self.config.continuous_min
        high = self.config.continuous_max
        clamped = continuous_actions.clamp(min=low, max=high)
        normalized = (clamped - low) / (high - low)
        continuous_bins = torch.round(normalized * (self.config.action_bins - 1)).long()

        terminate = terminate.long().clamp(min=0, max=1).unsqueeze(-1)
        raw_bins = torch.cat([terminate, continuous_bins], dim=-1)
        return raw_bins + self.config.action_token_offset

    def decode(self, action_token_ids: Tensor) -> dict[str, Tensor]:
        """Decodes token IDs back to approximate actions."""

        raw_bins = action_token_ids.long() - self.config.action_token_offset
        raw_bins = raw_bins.clamp(min=0, max=self.config.action_bins - 1)

        terminate = raw_bins[:, 0].clamp(max=1)
        continuous_bins = raw_bins[:, 1:].float()

        low = self.config.continuous_min
        high = self.config.continuous_max
        continuous = continuous_bins / (self.config.action_bins - 1)
        continuous = continuous * (high - low) + low

        return {
            "terminate": terminate,
            "continuous": continuous,
            "raw_bins": raw_bins,
        }

    def valid_token_mask(self, vocab_size: int, *, position_index: int, device: torch.device) -> Tensor:
        """Returns the valid token mask for one action position."""

        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        start = self.config.action_token_offset
        end = start + self.config.action_bins
        mask[start:end] = True

        # The first action token is the termination flag, which only uses IDs 0/1
        # inside the action token range.
        if position_index == 0:
            mask[:] = False
            mask[start : start + 2] = True
        return mask

    def constrain_logits(self, logits: Tensor, *, position_index: int) -> Tensor:
        """Masks logits to valid action tokens for one decoding position."""

        mask = self.valid_token_mask(
            logits.size(-1),
            position_index=position_index,
            device=logits.device,
        )
        invalid_mask = ~mask[None, :]
        return logits.masked_fill(invalid_mask, torch.finfo(logits.dtype).min)


class RT2Model(nn.Module):
    """Educational RT-2 model with two approximated VLM backbone modes."""

    def __init__(self, config: RT2Config) -> None:
        super().__init__()
        self.config = config
        self.action_tokenizer = RT2ActionTokenizer(config.action_config)

        text_cfg = config.text_config
        action_cfg = config.action_config
        model_width = config.transformer_config.width

        if action_cfg.action_token_offset + action_cfg.action_bins > text_cfg.vocab_size:
            raise ValueError(
                "The reserved action-token range must fit inside the vocabulary. "
                f"Got offset={action_cfg.action_token_offset}, bins={action_cfg.action_bins}, "
                f"vocab_size={text_cfg.vocab_size}."
            )

        self.vision_encoder = RT2VisionEncoder(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.width, model_width, bias=False)

        self.token_embedding = nn.Embedding(text_cfg.vocab_size, model_width)
        self.modality_embedding = nn.Embedding(3, model_width)
        self.lm_head = nn.Linear(model_width, text_cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        backbone_type = config.transformer_config.backbone_type
        if backbone_type == "encoder_decoder":
            self.backbone = RT2EncoderDecoderBackbone(config)
        elif backbone_type == "decoder_only":
            self.backbone = RT2DecoderOnlyBackbone(config)
        else:
            raise ValueError(
                'backbone_type must be either "encoder_decoder" or "decoder_only", '
                f"got {backbone_type!r}."
            )

    def _normalize_images(self, images: Tensor) -> Tensor:
        """Normalizes image input shape to ``[batch, num_images, C, H, W]``."""

        if images.ndim == 4:
            return images.unsqueeze(1)
        if images.ndim == 5:
            return images
        raise ValueError(
            "images must have shape [batch, C, H, W] or [batch, num_images, C, H, W], "
            f"got {tuple(images.shape)}."
        )

    def _prompt_mask(self, prompt_ids: Tensor, prompt_attention_mask: Tensor | None) -> Tensor:
        """Builds a boolean prompt mask."""

        if prompt_attention_mask is not None:
            return prompt_attention_mask.bool()
        return prompt_ids.ne(self.config.text_config.pad_token_id)

    def _target_mask(self, target_ids: Tensor, target_attention_mask: Tensor | None) -> Tensor:
        """Builds a boolean target mask."""

        if target_attention_mask is not None:
            return target_attention_mask.bool()
        return target_ids.ne(self.config.text_config.pad_token_id)

    def _shift_targets_right(self, target_ids: Tensor) -> Tensor:
        """Creates teacher-forced decoder inputs from target IDs."""

        batch_size, target_length = target_ids.shape
        bos = torch.full(
            (batch_size, 1),
            self.config.text_config.bos_token_id,
            dtype=target_ids.dtype,
            device=target_ids.device,
        )
        return torch.cat([bos, target_ids[:, :-1]], dim=1)

    def _encode_visual_tokens(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes images and returns projected tokens plus their mask."""

        images = self._normalize_images(images)
        visual_tokens = self.vision_encoder(images)
        visual_tokens = self.visual_projection(visual_tokens)
        visual_tokens = visual_tokens + self.modality_embedding.weight[0][None, None, :]
        visual_mask = torch.ones(
            visual_tokens.size(0),
            visual_tokens.size(1),
            device=visual_tokens.device,
            dtype=torch.bool,
        )
        return visual_tokens, visual_mask

    def _embed_prompt_tokens(self, prompt_ids: Tensor) -> Tensor:
        """Embeds instruction or web prompt text."""

        prompt_embeddings = self.token_embedding(prompt_ids)
        prompt_embeddings = prompt_embeddings + self.modality_embedding.weight[1][None, None, :]
        return prompt_embeddings

    def _embed_target_tokens(self, token_ids: Tensor) -> Tensor:
        """Embeds teacher-forced target inputs."""

        token_embeddings = self.token_embedding(token_ids)
        token_embeddings = token_embeddings + self.modality_embedding.weight[2][None, None, :]
        return token_embeddings

    def _masked_lm_loss(self, logits: Tensor, target_ids: Tensor, target_mask: Tensor) -> Tensor:
        """Computes the masked next-token loss."""

        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        )
        flat_mask = target_mask.reshape(-1).to(flat_loss.dtype)
        return (flat_loss * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)

    def forward(
        self,
        images: Tensor,
        prompt_ids: Tensor,
        *,
        target_ids: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
    ) -> RT2Output:
        """Runs one training-style forward pass.

        Args:
            images: Input image tensor.
            prompt_ids: Prompt token IDs describing the instruction or VQA task.
            target_ids: Optional target token IDs. These can be robot-action
                tokens or ordinary natural-language tokens.
            prompt_attention_mask: Optional prompt mask.
            target_attention_mask: Optional target mask.

        Returns:
            An ``RT2Output`` object.
        """

        visual_tokens, visual_mask = self._encode_visual_tokens(images)
        prompt_mask = self._prompt_mask(prompt_ids, prompt_attention_mask)
        prompt_embeddings = self._embed_prompt_tokens(prompt_ids)

        backbone_type = self.config.transformer_config.backbone_type

        if backbone_type == "encoder_decoder":
            context_embeddings = torch.cat([visual_tokens, prompt_embeddings], dim=1)
            context_mask = torch.cat([visual_mask, prompt_mask], dim=1)

            if target_ids is None:
                return RT2Output(
                    visual_tokens=visual_tokens,
                    prompt_embeddings=prompt_embeddings,
                    backbone_inputs=context_embeddings,
                    decoder_input_ids=None,
                    logits=None,
                    loss=None,
                )

            target_mask = self._target_mask(target_ids, target_attention_mask)
            decoder_input_ids = self._shift_targets_right(target_ids)
            decoder_embeddings = self._embed_target_tokens(decoder_input_ids)

            _, decoded_states = self.backbone(
                context_embeddings,
                context_mask=context_mask,
                decoder_embeddings=decoder_embeddings,
                decoder_mask=target_mask,
            )
            logits = self.lm_head(decoded_states)
            loss = self._masked_lm_loss(logits, target_ids, target_mask)
            return RT2Output(
                visual_tokens=visual_tokens,
                prompt_embeddings=prompt_embeddings,
                backbone_inputs=context_embeddings,
                decoder_input_ids=decoder_input_ids,
                logits=logits,
                loss=loss,
            )

        target_mask = None
        decoder_input_ids = None

        if target_ids is not None:
            target_mask = self._target_mask(target_ids, target_attention_mask)
            decoder_input_ids = self._shift_targets_right(target_ids)
            decoder_embeddings = self._embed_target_tokens(decoder_input_ids)
            sequence_embeddings = torch.cat([visual_tokens, prompt_embeddings, decoder_embeddings], dim=1)
            sequence_mask = torch.cat([visual_mask, prompt_mask, target_mask], dim=1)
        else:
            sequence_embeddings = torch.cat([visual_tokens, prompt_embeddings], dim=1)
            sequence_mask = torch.cat([visual_mask, prompt_mask], dim=1)

        hidden_states = self.backbone(sequence_embeddings, sequence_mask=sequence_mask)

        if target_ids is None:
            return RT2Output(
                visual_tokens=visual_tokens,
                prompt_embeddings=prompt_embeddings,
                backbone_inputs=sequence_embeddings,
                decoder_input_ids=None,
                logits=None,
                loss=None,
            )

        logits = self.lm_head(hidden_states[:, -target_ids.size(1) :])
        loss = self._masked_lm_loss(logits, target_ids, target_mask)
        return RT2Output(
            visual_tokens=visual_tokens,
            prompt_embeddings=prompt_embeddings,
            backbone_inputs=sequence_embeddings,
            decoder_input_ids=decoder_input_ids,
            logits=logits,
            loss=loss,
        )

    @torch.no_grad()
    def generate_action_tokens(
        self,
        images: Tensor,
        prompt_ids: Tensor,
        *,
        prompt_attention_mask: Tensor | None = None,
        max_action_tokens: int | None = None,
    ) -> Tensor:
        """Greedily decodes one RT-2 robot action sequence.

        Args:
            images: Input images.
            prompt_ids: Instruction or robot-action prompt tokens.
            prompt_attention_mask: Optional prompt mask.
            max_action_tokens: Number of action tokens to generate. Defaults to
                the configured RT-2 action sequence length.

        Returns:
            Tensor with shape ``[batch, action_sequence_length]``.
        """

        visual_tokens, visual_mask = self._encode_visual_tokens(images)
        prompt_mask = self._prompt_mask(prompt_ids, prompt_attention_mask)
        prompt_embeddings = self._embed_prompt_tokens(prompt_ids)
        step_count = max_action_tokens or self.config.action_config.action_sequence_length

        if self.config.transformer_config.backbone_type == "encoder_decoder":
            context_embeddings = torch.cat([visual_tokens, prompt_embeddings], dim=1)
            context_mask = torch.cat([visual_mask, prompt_mask], dim=1)

            generated = torch.full(
                (prompt_ids.size(0), 1),
                self.config.text_config.bos_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            outputs: list[Tensor] = []

            for position_index in range(step_count):
                decoder_embeddings = self._embed_target_tokens(generated)
                decoder_mask = torch.ones_like(generated, dtype=torch.bool)
                _, decoded_states = self.backbone(
                    context_embeddings,
                    context_mask=context_mask,
                    decoder_embeddings=decoder_embeddings,
                    decoder_mask=decoder_mask,
                )
                next_logits = self.lm_head(decoded_states[:, -1])
                next_logits = self.action_tokenizer.constrain_logits(
                    next_logits,
                    position_index=position_index,
                )
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                outputs.append(next_token)
                generated = torch.cat([generated, next_token], dim=1)

            return torch.cat(outputs, dim=1)

        generated = torch.full(
            (prompt_ids.size(0), 1),
            self.config.text_config.bos_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        outputs = []

        for position_index in range(step_count):
            decoder_embeddings = self._embed_target_tokens(generated)
            sequence_embeddings = torch.cat([visual_tokens, prompt_embeddings, decoder_embeddings], dim=1)
            generated_mask = torch.ones_like(generated, dtype=torch.bool)
            sequence_mask = torch.cat([visual_mask, prompt_mask, generated_mask], dim=1)

            hidden_states = self.backbone(sequence_embeddings, sequence_mask=sequence_mask)
            next_logits = self.lm_head(hidden_states[:, -1])
            next_logits = self.action_tokenizer.constrain_logits(
                next_logits,
                position_index=position_index,
            )
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            outputs.append(next_token)
            generated = torch.cat([generated, next_token], dim=1)

        return torch.cat(outputs, dim=1)

    def decode_action_tokens(self, action_token_ids: Tensor) -> dict[str, Tensor]:
        """Convenience wrapper around the RT-2 action tokenizer."""

        return self.action_tokenizer.decode(action_token_ids)

    def co_finetune_step(
        self,
        *,
        robot_images: Tensor,
        robot_prompt_ids: Tensor,
        robot_target_ids: Tensor,
        web_images: Tensor,
        web_prompt_ids: Tensor,
        web_target_ids: Tensor,
        robot_weight: float = 1.0,
        web_weight: float = 1.0,
    ) -> RT2CoFineTuneOutput:
        """Computes the mixed loss used to illustrate RT-2 co-fine-tuning.

        The RT-2 paper emphasizes that the model is not fine-tuned on robot data
        alone. Instead, robot trajectories and original web-style VLM tasks are
        both included during training. This helper mirrors that recipe with two
        separate forward passes and a weighted loss sum.
        """

        robot_output = self(
            robot_images,
            robot_prompt_ids,
            target_ids=robot_target_ids,
        )
        web_output = self(
            web_images,
            web_prompt_ids,
            target_ids=web_target_ids,
        )

        if robot_output.loss is None or web_output.loss is None:
            raise RuntimeError("Both robot and web passes must produce valid losses.")

        loss = robot_weight * robot_output.loss + web_weight * web_output.loss
        return RT2CoFineTuneOutput(
            robot_output=robot_output,
            web_output=web_output,
            loss=loss,
        )


def build_rt2_pali_x_tiny() -> RT2Model:
    """Builds a tiny encoder-decoder approximation of RT-2-PaLI-X."""

    config = RT2Config(
        vision_config=RT2VisionConfig(
            image_size=64,
            patch_size=16,
            width=64,
            layers=2,
            heads=4,
            max_images=2,
        ),
        text_config=RT2TextConfig(
            vocab_size=512,
            max_prompt_tokens=16,
            max_target_tokens=16,
        ),
        action_config=RT2ActionConfig(action_token_offset=256),
        transformer_config=RT2TransformerConfig(
            width=128,
            encoder_layers=2,
            decoder_layers=2,
            heads=4,
            backbone_type="encoder_decoder",
        ),
    )
    return RT2Model(config)


def build_rt2_palm_e_tiny() -> RT2Model:
    """Builds a tiny decoder-only approximation of RT-2-PaLM-E."""

    config = RT2Config(
        vision_config=RT2VisionConfig(
            image_size=64,
            patch_size=16,
            width=64,
            layers=2,
            heads=4,
            max_images=2,
        ),
        text_config=RT2TextConfig(
            vocab_size=512,
            max_prompt_tokens=16,
            max_target_tokens=16,
        ),
        action_config=RT2ActionConfig(action_token_offset=256),
        transformer_config=RT2TransformerConfig(
            width=128,
            encoder_layers=0,
            decoder_layers=3,
            heads=4,
            backbone_type="decoder_only",
        ),
    )
    return RT2Model(config)


def _sample_robot_action_targets(model: RT2Model, batch_size: int) -> Tensor:
    """Creates a tiny batch of synthetic RT-2 robot-action targets."""

    continuous_actions = torch.empty(batch_size, 7).uniform_(-1.0, 1.0)
    terminate = torch.randint(0, 2, (batch_size,))
    return model.action_tokenizer.encode(continuous_actions, terminate)


def _smoke_test() -> None:
    """Runs small forward passes for both RT-2 backbone modes."""

    torch.manual_seed(0)

    for name, builder in (
        ("RT2-PaLI-X-tiny", build_rt2_pali_x_tiny),
        ("RT2-PaLM-E-tiny", build_rt2_palm_e_tiny),
    ):
        model = builder()
        model.eval()

        images = torch.randn(2, 2, 3, 64, 64)
        prompt_ids = torch.randint(3, 120, (2, 10))
        robot_target_ids = _sample_robot_action_targets(model, batch_size=2)
        web_target_ids = torch.randint(3, 200, (2, 12))

        robot_output = model(images, prompt_ids, target_ids=robot_target_ids)
        web_output = model(images, prompt_ids, target_ids=web_target_ids)
        combined = model.co_finetune_step(
            robot_images=images,
            robot_prompt_ids=prompt_ids,
            robot_target_ids=robot_target_ids,
            web_images=images,
            web_prompt_ids=prompt_ids,
            web_target_ids=web_target_ids,
            robot_weight=2.0,
            web_weight=1.0,
        )
        generated = model.generate_action_tokens(images, prompt_ids)
        decoded = model.decode_action_tokens(generated)

        print(name)
        print("visual_tokens:", tuple(robot_output.visual_tokens.shape))
        print("backbone_inputs:", tuple(robot_output.backbone_inputs.shape))
        print("robot_logits:", tuple(robot_output.logits.shape) if robot_output.logits is not None else None)
        print("web_logits:", tuple(web_output.logits.shape) if web_output.logits is not None else None)
        print("generated_action_tokens:", tuple(generated.shape))
        print("decoded_continuous:", tuple(decoded["continuous"].shape))
        print("robot_loss:", float(robot_output.loss.detach()) if robot_output.loss is not None else None)
        print("web_loss:", float(web_output.loss.detach()) if web_output.loss is not None else None)
        print("co_finetune_loss:", float(combined.loss.detach()))
        print("---")


if __name__ == "__main__":
    _smoke_test()
