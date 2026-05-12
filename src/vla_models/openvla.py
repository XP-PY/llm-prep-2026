"""A compact educational OpenVLA implementation in PyTorch.

This module follows the architecture summary in ``docs/VLAs/OpenVLA.md``:

1. A fused DINOv2 + SigLIP visual tokenizer encodes one robot observation.
2. A small two-layer projector maps visual patch features to language width.
3. A decoder-only language backbone predicts robot actions as short token
   sequences.
4. Each action dimension is discretized independently into 256 bins using
   per-dimension quantile ranges.
5. Practical adaptation modes such as frozen-vision fine-tuning, sandwich
   tuning, and LoRA are exposed as explicit utilities.

This implementation is intentionally educational rather than checkpoint-faithful:

* The visual towers are compact ViT-style encoders, not real DINOv2/SigLIP
  checkpoints.
* The language backbone is a small Llama-like decoder stack rather than the
  production Llama 2 7B backbone used by OpenVLA.
* LoRA is implemented as a readable in-place adapter utility rather than a
  full trainer-integrated fine-tuning stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class OpenVLAVisionConfig:
    """Configuration for the educational fused visual tokenizer.

    Attributes:
        image_size: Square input resolution expected by the compact towers.
        in_channels: Number of input image channels.
        patch_size: Patch size used by both ViT-style visual stems.
        siglip_width: Hidden width of the SigLIP-like branch.
        dino_width: Hidden width of the DINOv2-like branch.
        branch_layers: Number of self-attention blocks in each branch.
        branch_heads: Number of attention heads in each visual branch.
        mlp_ratio: Expansion ratio used inside branch feed-forward blocks.
        dropout: Dropout applied in the visual branches.
        use_dino: Whether to fuse the DINO-style branch with the SigLIP branch.
    """

    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 14
    siglip_width: int = 512
    dino_width: int = 512
    branch_layers: int = 6
    branch_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_dino: bool = True

    @property
    def patches_per_image(self) -> int:
        """Returns the number of patch tokens emitted per observation."""

        grid_size = self.image_size // self.patch_size
        return grid_size * grid_size

    @property
    def fused_width(self) -> int:
        """Returns the concatenated feature width after branch fusion."""

        if self.use_dino:
            return self.siglip_width + self.dino_width
        return self.siglip_width


@dataclass(frozen=True)
class OpenVLATextConfig:
    """Configuration for prompts and action-token decoding."""

    vocab_size: int = 32_000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    max_prompt_tokens: int = 64
    max_action_tokens: int = 7


@dataclass(frozen=True)
class OpenVLAActionConfig:
    """Configuration for OpenVLA-style 7D action tokenization.

    The paper discretizes each of the seven action dimensions into 256 bins and
    reserves a short token sequence of length seven for every control step.
    Each dimension uses its own 1% / 99% quantile range.
    """

    action_token_offset: int = 31_744
    action_bins: int = 256
    quantile_lows: tuple[float, ...] = (-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, -1.0)
    quantile_highs: tuple[float, ...] = (0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0)

    def __post_init__(self) -> None:
        if len(self.quantile_lows) != 7 or len(self.quantile_highs) != 7:
            raise ValueError("OpenVLA expects exactly seven action dimensions.")

    @property
    def action_names(self) -> tuple[str, ...]:
        """Returns the ordered 7D action names used by the tokenizer."""

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
    def action_sequence_length(self) -> int:
        """Returns the number of action tokens emitted per action step."""

        return len(self.action_names)

    @property
    def action_token_end(self) -> int:
        """Returns the exclusive upper bound of the reserved action range."""

        return self.action_token_offset + self.action_bins


@dataclass(frozen=True)
class OpenVLABackboneConfig:
    """Configuration for the decoder-only language backbone."""

    width: int = 512
    layers: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0


@dataclass(frozen=True)
class OpenVLAConfig:
    """Top-level configuration for the educational OpenVLA model."""

    vision_config: OpenVLAVisionConfig = field(default_factory=OpenVLAVisionConfig)
    text_config: OpenVLATextConfig = field(default_factory=OpenVLATextConfig)
    action_config: OpenVLAActionConfig = field(default_factory=OpenVLAActionConfig)
    backbone_config: OpenVLABackboneConfig = field(default_factory=OpenVLABackboneConfig)


@dataclass
class OpenVLAOutput:
    """Container returned by ``OpenVLAModel.forward``.

    Attributes:
        siglip_tokens: Raw SigLIP-like patch tokens with shape ``[B, P, D_s]``.
        dino_tokens: Optional DINO-like patch tokens with shape ``[B, P, D_d]``.
        fused_visual_tokens: Channel-concatenated visual patch tokens before the
            projector.
        projected_visual_tokens: Visual tokens after the 2-layer projector.
        prompt_embeddings: Embedded prompt tokens before action tokens are
            appended.
        backbone_inputs: Full sequence passed into the decoder-only backbone.
        decoder_input_ids: Teacher-forced input action IDs used during training.
        logits: Action-token logits aligned with the target sequence.
        loss: Optional action-only next-token cross-entropy.
    """

    siglip_tokens: Tensor
    dino_tokens: Tensor | None
    fused_visual_tokens: Tensor
    projected_visual_tokens: Tensor
    prompt_embeddings: Tensor
    backbone_inputs: Tensor
    decoder_input_ids: Tensor | None
    logits: Tensor | None
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    """Root-mean-square normalization used in the backbone and visual towers."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class FeedForward(nn.Module):
    """Minimal gated-free feed-forward block used in the compact transformers."""

    def __init__(self, width: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_width = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden_width, bias=False)
        self.fc2 = nn.Linear(hidden_width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.gelu(self.fc1(hidden_states), approximate="tanh")
        hidden_states = self.dropout(hidden_states)
        return self.fc2(hidden_states)


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with optional causal masking."""

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

    def _reshape_heads(self, hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return hidden_states.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        query = self._reshape_heads(self.q_proj(hidden_states))
        key = self._reshape_heads(self.k_proj(hidden_states))
        value = self._reshape_heads(self.v_proj(hidden_states))

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        seq_len = hidden_states.size(1)

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=1,
            )
            attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = ~attention_mask[:, None, None, :]
            attention_scores = attention_scores.masked_fill(key_mask, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(hidden_states.size(0), seq_len, self.width)
        return self.out_proj(attention_output)


class TransformerBlock(nn.Module):
    """A small pre-norm transformer block shared by vision and language stacks."""

    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.attn = MultiHeadAttention(width, heads, dropout=dropout)
        self.ffn_norm = RMSNorm(width)
        self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            causal=causal,
        )
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class PatchEmbed(nn.Module):
    """Patchifies one image into a patch-token sequence."""

    def __init__(self, config: OpenVLAVisionConfig, width: int) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.grid_size = config.image_size // config.patch_size
        self.proj = nn.Conv2d(
            config.in_channels,
            width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, images: Tensor) -> Tensor:
        patch_grid = self.proj(images)
        patch_grid = patch_grid.flatten(2).transpose(1, 2)
        return patch_grid


class VisionBranchEncoder(nn.Module):
    """A compact ViT-style branch that approximates one visual backbone."""

    def __init__(self, config: OpenVLAVisionConfig, width: int) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(config, width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.patches_per_image, width)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    width=width,
                    heads=config.branch_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.branch_layers)
            ]
        )
        self.norm = RMSNorm(width)

    def forward(self, images: Tensor) -> Tensor:
        hidden_states = self.patch_embed(images) + self.position_embedding
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.norm(hidden_states)


class DualVisionEncoder(nn.Module):
    """Fuses SigLIP-like and DINO-like patch tokens channel-wise."""

    def __init__(self, config: OpenVLAVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.siglip_branch = VisionBranchEncoder(config, config.siglip_width)
        self.dino_branch = (
            VisionBranchEncoder(config, config.dino_width) if config.use_dino else None
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor | None, Tensor]:
        if images.dim() == 5:
            if images.size(1) != 1:
                raise ValueError("Educational OpenVLA expects exactly one image per prompt.")
            images = images[:, 0]
        if images.dim() != 4:
            raise ValueError("images must have shape [B, C, H, W] or [B, 1, C, H, W].")

        siglip_tokens = self.siglip_branch(images)
        dino_tokens = self.dino_branch(images) if self.dino_branch is not None else None

        if dino_tokens is None:
            fused_tokens = siglip_tokens
        else:
            fused_tokens = torch.cat([siglip_tokens, dino_tokens], dim=-1)

        return siglip_tokens, dino_tokens, fused_tokens


class TwoLayerProjector(nn.Module):
    """The small MLP projector used between vision tokens and language width."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_width, output_width, bias=False)
        self.fc2 = nn.Linear(output_width, output_width, bias=False)
        self.norm = RMSNorm(output_width)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        visual_tokens = F.gelu(self.fc1(visual_tokens), approximate="tanh")
        visual_tokens = self.fc2(visual_tokens)
        return self.norm(visual_tokens)


class OpenVLAActionTokenizer:
    """Quantizes a 7D control vector into seven action tokens.

    The tokenizer mirrors the paper's per-dimension quantile binning:

    * each action dimension uses its own ``[Q_0.01, Q_0.99]`` range
    * each dimension is discretized into 256 bins
    * de-tokenization maps a predicted bin back to the center of its interval
    """

    def __init__(self, config: OpenVLAActionConfig) -> None:
        self.config = config
        self._lows = torch.tensor(config.quantile_lows, dtype=torch.float32)
        self._highs = torch.tensor(config.quantile_highs, dtype=torch.float32)

    def _to_device(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        lows = self._lows.to(reference.device, reference.dtype)
        highs = self._highs.to(reference.device, reference.dtype)
        return lows, highs

    def encode(self, continuous_actions: Tensor) -> Tensor:
        """Converts continuous actions to token IDs with shape ``[..., 7]``."""

        if continuous_actions.size(-1) != self.config.action_sequence_length:
            raise ValueError("continuous_actions must end with seven OpenVLA dimensions.")

        lows, highs = self._to_device(continuous_actions)
        bin_widths = (highs - lows) / float(self.config.action_bins)
        clipped = continuous_actions.clamp(min=lows, max=highs)
        bins = torch.floor((clipped - lows) / bin_widths).long()
        bins = bins.clamp(min=0, max=self.config.action_bins - 1)
        return bins + self.config.action_token_offset

    def decode(self, token_ids: Tensor) -> dict[str, Tensor]:
        """Maps action token IDs back to continuous control values."""

        if token_ids.size(-1) != self.config.action_sequence_length:
            raise ValueError("token_ids must end with seven OpenVLA action tokens.")

        bins = (token_ids - self.config.action_token_offset).clamp(
            min=0,
            max=self.config.action_bins - 1,
        )
        lows, highs = self._to_device(token_ids.float())
        bin_widths = (highs - lows) / float(self.config.action_bins)
        continuous = lows + (bins.float() + 0.5) * bin_widths
        return {
            name: continuous[..., index]
            for index, name in enumerate(self.config.action_names)
        }

    def mask_non_action_logits(self, logits: Tensor) -> Tensor:
        """Masks all vocabulary positions outside the reserved action range."""

        masked_logits = torch.full_like(logits, float("-inf"))
        start = self.config.action_token_offset
        end = self.config.action_token_end
        masked_logits[..., start:end] = logits[..., start:end]
        return masked_logits


class OpenVLADecoderOnlyBackbone(nn.Module):
    """A small Llama-like decoder that consumes one fused token stream."""

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__()
        text_config = config.text_config
        vision_config = config.vision_config
        backbone_config = config.backbone_config

        self.max_sequence_length = (
            vision_config.patches_per_image
            + text_config.max_prompt_tokens
            + text_config.max_action_tokens
        )
        self.position_embedding = nn.Embedding(self.max_sequence_length, backbone_config.width)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    width=backbone_config.width,
                    heads=backbone_config.heads,
                    mlp_ratio=backbone_config.mlp_ratio,
                    dropout=backbone_config.dropout,
                )
                for _ in range(backbone_config.layers)
            ]
        )
        self.norm = RMSNorm(backbone_config.width)

    def forward(self, hidden_states: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        seq_len = hidden_states.size(1)
        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_sequence_length={self.max_sequence_length}."
            )

        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        hidden_states = hidden_states + self.position_embedding(position_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, causal=True)

        return self.norm(hidden_states)


class LoRALinear(nn.Module):
    """A readable LoRA wrapper around a frozen linear layer."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LoRA.")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank

        self.base = nn.Linear(self.in_features, self.out_features, bias=base_layer.bias is not None)
        self.base.load_state_dict(base_layer.state_dict())
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_a = nn.Linear(self.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.base(hidden_states) + self.lora_b(self.lora_a(hidden_states)) * self.scaling


def _replace_linear_with_lora(module: nn.Module, *, rank: int, alpha: float | None) -> None:
    """Recursively replaces linear layers inside ``module`` with ``LoRALinear``."""

    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            _replace_linear_with_lora(child, rank=rank, alpha=alpha)


class OpenVLAModel(nn.Module):
    """Educational OpenVLA model with decoder-only action prediction."""

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.action_tokenizer = OpenVLAActionTokenizer(config.action_config)

        self.vision_encoder = DualVisionEncoder(config.vision_config)
        self.projector = TwoLayerProjector(
            config.vision_config.fused_width,
            config.backbone_config.width,
        )
        self.token_embedding = nn.Embedding(
            config.text_config.vocab_size,
            config.backbone_config.width,
        )
        self.modality_embedding = nn.Embedding(3, config.backbone_config.width)
        self.backbone = OpenVLADecoderOnlyBackbone(config)
        self.lm_head = nn.Linear(
            config.backbone_config.width,
            config.text_config.vocab_size,
            bias=False,
        )

    def _visual_attention_mask(self, batch_size: int, device: torch.device) -> Tensor:
        visual_tokens = self.config.vision_config.patches_per_image
        return torch.ones(batch_size, visual_tokens, device=device, dtype=torch.bool)

    def _prompt_attention_mask(self, prompt_ids: Tensor, prompt_attention_mask: Tensor | None) -> Tensor:
        if prompt_attention_mask is not None:
            return prompt_attention_mask.bool()
        return prompt_ids.ne(self.config.text_config.pad_token_id)

    def _action_attention_mask(
        self,
        action_ids: Tensor,
        action_attention_mask: Tensor | None,
    ) -> Tensor:
        if action_attention_mask is not None:
            return action_attention_mask.bool()
        return action_ids.ne(self.config.text_config.pad_token_id)

    def _shift_targets_right(self, target_action_ids: Tensor) -> Tensor:
        batch_size, sequence_length = target_action_ids.shape
        shifted = torch.full_like(target_action_ids, self.config.text_config.pad_token_id)
        shifted[:, 0] = self.config.text_config.bos_token_id
        if sequence_length > 1:
            shifted[:, 1:] = target_action_ids[:, :-1]
        return shifted

    def _encode_visual_tokens(self, images: Tensor) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        siglip_tokens, dino_tokens, fused_visual_tokens = self.vision_encoder(images)
        projected_visual_tokens = self.projector(fused_visual_tokens)
        projected_visual_tokens = projected_visual_tokens + self.modality_embedding.weight[0]
        return siglip_tokens, dino_tokens, fused_visual_tokens, projected_visual_tokens

    def _embed_prompt_tokens(self, prompt_ids: Tensor) -> Tensor:
        prompt_embeddings = self.token_embedding(prompt_ids)
        return prompt_embeddings + self.modality_embedding.weight[1]

    def _embed_action_tokens(self, action_ids: Tensor) -> Tensor:
        action_embeddings = self.token_embedding(action_ids)
        return action_embeddings + self.modality_embedding.weight[2]

    def _masked_action_loss(
        self,
        logits: Tensor,
        target_action_ids: Tensor,
        target_action_mask: Tensor,
    ) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(dim=-1, index=target_action_ids.unsqueeze(-1)).squeeze(-1)
        target_action_mask = target_action_mask.float()
        loss = -(gathered * target_action_mask).sum() / target_action_mask.sum().clamp_min(1.0)
        return loss

    def _forward_with_action_prefix(
        self,
        projected_visual_tokens: Tensor,
        prompt_embeddings: Tensor,
        prompt_mask: Tensor,
        action_prefix_ids: Tensor,
        action_prefix_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        action_embeddings = self._embed_action_tokens(action_prefix_ids)
        backbone_inputs = torch.cat(
            [projected_visual_tokens, prompt_embeddings, action_embeddings],
            dim=1,
        )
        visual_mask = self._visual_attention_mask(
            batch_size=projected_visual_tokens.size(0),
            device=projected_visual_tokens.device,
        )
        attention_mask = torch.cat([visual_mask, prompt_mask, action_prefix_mask], dim=1)
        hidden_states = self.backbone(backbone_inputs, attention_mask=attention_mask)
        return hidden_states, backbone_inputs

    def forward(
        self,
        images: Tensor,
        prompt_ids: Tensor,
        *,
        target_action_ids: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        target_action_mask: Tensor | None = None,
    ) -> OpenVLAOutput:
        """Runs an OpenVLA forward pass.

        When ``target_action_ids`` are provided, the model follows teacher
        forcing and computes the action-only cross-entropy described in the
        paper. Without targets, the method still returns multimodal prefix
        states, which are then reused by ``generate_action_tokens``.
        """

        (
            siglip_tokens,
            dino_tokens,
            fused_visual_tokens,
            projected_visual_tokens,
        ) = self._encode_visual_tokens(images)
        prompt_embeddings = self._embed_prompt_tokens(prompt_ids)
        prompt_mask = self._prompt_attention_mask(prompt_ids, prompt_attention_mask)

        if target_action_ids is None:
            visual_mask = self._visual_attention_mask(
                batch_size=projected_visual_tokens.size(0),
                device=projected_visual_tokens.device,
            )
            backbone_inputs = torch.cat([projected_visual_tokens, prompt_embeddings], dim=1)
            attention_mask = torch.cat([visual_mask, prompt_mask], dim=1)
            self.backbone(backbone_inputs, attention_mask=attention_mask)
            return OpenVLAOutput(
                siglip_tokens=siglip_tokens,
                dino_tokens=dino_tokens,
                fused_visual_tokens=fused_visual_tokens,
                projected_visual_tokens=projected_visual_tokens,
                prompt_embeddings=prompt_embeddings,
                backbone_inputs=backbone_inputs,
                decoder_input_ids=None,
                logits=None,
                loss=None,
            )

        decoder_input_ids = self._shift_targets_right(target_action_ids)
        action_mask = self._action_attention_mask(target_action_ids, target_action_mask)
        hidden_states, backbone_inputs = self._forward_with_action_prefix(
            projected_visual_tokens,
            prompt_embeddings,
            prompt_mask,
            decoder_input_ids,
            action_mask,
        )
        action_hidden_states = hidden_states[:, -decoder_input_ids.size(1) :, :]
        logits = self.lm_head(action_hidden_states)
        loss = self._masked_action_loss(logits, target_action_ids, action_mask)

        return OpenVLAOutput(
            siglip_tokens=siglip_tokens,
            dino_tokens=dino_tokens,
            fused_visual_tokens=fused_visual_tokens,
            projected_visual_tokens=projected_visual_tokens,
            prompt_embeddings=prompt_embeddings,
            backbone_inputs=backbone_inputs,
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
        """Greedily decodes the seven OpenVLA action tokens."""

        (
            _siglip_tokens,
            _dino_tokens,
            _fused_visual_tokens,
            projected_visual_tokens,
        ) = self._encode_visual_tokens(images)
        prompt_embeddings = self._embed_prompt_tokens(prompt_ids)
        prompt_mask = self._prompt_attention_mask(prompt_ids, prompt_attention_mask)

        batch_size = prompt_ids.size(0)
        steps = max_action_tokens or self.config.action_config.action_sequence_length
        generated_ids = torch.full(
            (batch_size, 1),
            fill_value=self.config.text_config.bos_token_id,
            device=prompt_ids.device,
            dtype=torch.long,
        )

        for _ in range(steps):
            prefix_mask = torch.ones_like(generated_ids, dtype=torch.bool)
            hidden_states, _ = self._forward_with_action_prefix(
                projected_visual_tokens,
                prompt_embeddings,
                prompt_mask,
                generated_ids,
                prefix_mask,
            )
            next_logits = self.lm_head(hidden_states[:, -1, :])
            next_logits = self.action_tokenizer.mask_non_action_logits(next_logits)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids[:, 1:]

    def decode_action_tokens(self, action_token_ids: Tensor) -> dict[str, Tensor]:
        """Maps generated action token IDs back to continuous control values."""

        return self.action_tokenizer.decode(action_token_ids)

    def format_robot_prompt(self, task_instruction: str) -> str:
        """Formats the robot-control prompt template used in the paper."""

        return f"What should the robot do to {task_instruction}? A:"

    def count_trainable_parameters(self) -> int:
        """Returns the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def set_finetuning_strategy(self, strategy: str) -> None:
        """Applies one of several practical OpenVLA-style tuning strategies."""

        normalized = strategy.lower()
        for parameter in self.parameters():
            parameter.requires_grad = False

        if normalized == "full":
            for parameter in self.parameters():
                parameter.requires_grad = True
            return

        if normalized == "last_layer_only":
            modules = [
                self.projector,
                self.token_embedding,
                self.modality_embedding,
                self.backbone.blocks[-1],
                self.backbone.norm,
                self.lm_head,
            ]
        elif normalized == "frozen_vision":
            for parameter in self.parameters():
                parameter.requires_grad = True
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False
            return
        elif normalized == "sandwich":
            modules = [
                self.vision_encoder,
                self.projector,
                self.token_embedding,
                self.modality_embedding,
                self.backbone.blocks[-1],
                self.backbone.norm,
                self.lm_head,
            ]
        else:
            raise ValueError(
                "strategy must be one of: full, last_layer_only, frozen_vision, sandwich."
            )

        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def enable_lora(self, rank: int = 32, alpha: float | None = None, *, include_vision: bool = False) -> None:
        """Freezes the base model and injects LoRA into linear layers."""

        for parameter in self.parameters():
            parameter.requires_grad = False

        if include_vision:
            _replace_linear_with_lora(self.vision_encoder, rank=rank, alpha=alpha)
        _replace_linear_with_lora(self.projector, rank=rank, alpha=alpha)
        _replace_linear_with_lora(self.backbone, rank=rank, alpha=alpha)
        if isinstance(self.lm_head, nn.Linear):
            self.lm_head = LoRALinear(self.lm_head, rank=rank, alpha=alpha)


def build_openvla_tiny() -> OpenVLAModel:
    """Builds a tiny dual-vision OpenVLA for smoke tests and study."""

    config = OpenVLAConfig(
        vision_config=OpenVLAVisionConfig(
            image_size=64,
            patch_size=16,
            siglip_width=64,
            dino_width=64,
            branch_layers=2,
            branch_heads=4,
            use_dino=True,
        ),
        text_config=OpenVLATextConfig(
            vocab_size=512,
            max_prompt_tokens=16,
            max_action_tokens=7,
        ),
        action_config=OpenVLAActionConfig(
            action_token_offset=256,
            action_bins=256,
        ),
        backbone_config=OpenVLABackboneConfig(
            width=128,
            layers=4,
            heads=4,
        ),
    )
    return OpenVLAModel(config)


def build_openvla_siglip_only_tiny() -> OpenVLAModel:
    """Builds a tiny OpenVLA variant that keeps only the SigLIP-like branch."""

    config = OpenVLAConfig(
        vision_config=OpenVLAVisionConfig(
            image_size=64,
            patch_size=16,
            siglip_width=96,
            dino_width=64,
            branch_layers=2,
            branch_heads=4,
            use_dino=False,
        ),
        text_config=OpenVLATextConfig(
            vocab_size=512,
            max_prompt_tokens=16,
            max_action_tokens=7,
        ),
        action_config=OpenVLAActionConfig(
            action_token_offset=256,
            action_bins=256,
        ),
        backbone_config=OpenVLABackboneConfig(
            width=128,
            layers=4,
            heads=4,
        ),
    )
    return OpenVLAModel(config)


def _sample_action_targets(model: OpenVLAModel, batch_size: int) -> Tensor:
    """Samples tiny synthetic continuous actions and tokenizes them."""

    lows = torch.tensor(model.config.action_config.quantile_lows)
    highs = torch.tensor(model.config.action_config.quantile_highs)
    continuous = lows + torch.rand(batch_size, 7) * (highs - lows)
    return model.action_tokenizer.encode(continuous)


def _smoke_test() -> None:
    """Runs tiny training-style and inference-style checks."""

    torch.manual_seed(0)

    builders = [
        ("OpenVLA-tiny", build_openvla_tiny),
        ("OpenVLA-SigLIP-only-tiny", build_openvla_siglip_only_tiny),
    ]

    for name, builder in builders:
        model = builder()
        model.eval()

        images = torch.randn(2, 3, 64, 64)
        prompt_ids = torch.randint(3, 128, (2, 10))
        target_action_ids = _sample_action_targets(model, batch_size=2)

        output = model(images, prompt_ids, target_action_ids=target_action_ids)
        generated = model.generate_action_tokens(images, prompt_ids)
        decoded = model.decode_action_tokens(generated)

        print(name)
        print(f"  siglip_tokens: {tuple(output.siglip_tokens.shape)}")
        if output.dino_tokens is not None:
            print(f"  dino_tokens: {tuple(output.dino_tokens.shape)}")
        print(f"  fused_visual_tokens: {tuple(output.fused_visual_tokens.shape)}")
        print(f"  projected_visual_tokens: {tuple(output.projected_visual_tokens.shape)}")
        print(f"  logits: {tuple(output.logits.shape) if output.logits is not None else None}")
        print(f"  loss: {float(output.loss.detach()):.4f}")
        print(f"  generated_action_tokens: {tuple(generated.shape)}")
        print(f"  decoded_action_keys: {tuple(decoded.keys())}")

    strategy_model = build_openvla_tiny()
    strategy_model.set_finetuning_strategy("sandwich")
    print(f"sandwich_trainable_params: {strategy_model.count_trainable_parameters()}")

    lora_model = build_openvla_tiny()
    lora_model.enable_lora(rank=8, alpha=16.0)
    lora_model.eval()
    images = torch.randn(2, 3, 64, 64)
    prompt_ids = torch.randint(3, 128, (2, 10))
    target_action_ids = _sample_action_targets(lora_model, batch_size=2)
    lora_output = lora_model(images, prompt_ids, target_action_ids=target_action_ids)
    print(f"lora_trainable_params: {lora_model.count_trainable_parameters()}")
    print(f"lora_logits: {tuple(lora_output.logits.shape) if lora_output.logits is not None else None}")


if __name__ == "__main__":
    _smoke_test()
