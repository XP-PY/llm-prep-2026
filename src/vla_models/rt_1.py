"""A compact educational RT-1 implementation in PyTorch.

This module follows the architecture summary in ``docs/VLAs/RT_1.md``:

1. A language instruction is encoded into a compact sentence embedding.
2. Each image in a short history is processed by a FiLM-conditioned CNN inspired
   by the role of EfficientNet-B3 in the original paper.
3. The resulting ``9 x 9`` feature grid becomes 81 vision-language tokens.
4. TokenLearner compresses each frame from 81 tokens to 8 tokens.
5. A decoder-only Transformer reads the flattened history and predicts
   discretized robot action heads in parallel.

This code is intentionally educational rather than checkpoint-faithful:

* The image encoder is a FiLM-conditioned CNN, not an exact EfficientNet-B3.
* The instruction encoder is a compact embedding-pooler, not Universal Sentence
  Encoder.
* The policy predicts all action dimensions from one final action query token,
  matching the paper's non-autoregressive action decoding idea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RT1InstructionConfig:
    """Configuration for instruction tokenization and pooling.

    Attributes:
        vocab_size: Size of the instruction vocabulary.
        embed_dim: Token embedding width before pooling.
        sentence_dim: Final instruction embedding width used by FiLM.
        pad_token_id: Padding token ID used for masking.
    """

    vocab_size: int = 32_000
    embed_dim: int = 256
    sentence_dim: int = 512
    pad_token_id: int = 0


@dataclass(frozen=True)
class RT1VisionConfig:
    """Configuration for the FiLM-conditioned vision tokenizer.

    Attributes:
        image_size: Square input size used for each history frame.
        in_channels: Number of image channels.
        stem_width: Base channel width of the CNN tokenizer.
        token_width: Channel width of the final ``9 x 9`` token grid.
        token_grid_size: Spatial grid size before flattening to tokens.
        token_learner_tokens: Number of learned tokens output per frame.
        dropout: Dropout used inside the CNN tokenizer.
    """

    image_size: int = 300
    in_channels: int = 3
    stem_width: int = 128
    token_width: int = 512
    token_grid_size: int = 9
    token_learner_tokens: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class RT1TransformerConfig:
    """Configuration for the decoder-only Transformer backbone.

    Attributes:
        width: Hidden width of the policy transformer.
        layers: Number of decoder blocks.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio of the feed-forward block.
        dropout: Dropout used in attention and MLP layers.
        history_length: Number of image frames in the history window.
    """

    width: int = 512
    layers: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    history_length: int = 6


@dataclass(frozen=True)
class RT1ActionConfig:
    """Configuration for discretized action heads.

    The original RT-1 action space contains:

    * 7 arm dimensions
    * 3 base dimensions
    * 1 mode head with 3 classes

    Attributes:
        continuous_bins: Number of bins used for continuous action dimensions.
        mode_classes: Number of mode classes: arm, base, terminate.
    """

    continuous_bins: int = 256
    mode_classes: int = 3

    @property
    def head_specs(self) -> tuple[tuple[str, int], ...]:
        """Returns the action head names and class counts."""

        continuous_heads = (
            "arm_x",
            "arm_y",
            "arm_z",
            "arm_roll",
            "arm_pitch",
            "arm_yaw",
            "gripper",
            "base_x",
            "base_y",
            "base_yaw",
        )
        return tuple((name, self.continuous_bins) for name in continuous_heads) + (
            ("mode", self.mode_classes),
        )


@dataclass(frozen=True)
class RT1Config:
    """Top-level configuration for the educational RT-1 policy."""

    instruction_config: RT1InstructionConfig = field(default_factory=RT1InstructionConfig)
    vision_config: RT1VisionConfig = field(default_factory=RT1VisionConfig)
    transformer_config: RT1TransformerConfig = field(default_factory=RT1TransformerConfig)
    action_config: RT1ActionConfig = field(default_factory=RT1ActionConfig)


@dataclass
class RT1Output:
    """Output container returned by ``RT1Model.forward``.

    Attributes:
        instruction_embedding: Pooled instruction representation.
        raw_visual_tokens: Tokens before TokenLearner with shape
            ``[batch, history, 81, token_width]``.
        compressed_visual_tokens: Tokens after TokenLearner with shape
            ``[batch, history, 8, token_width]``.
        transformer_tokens: Token sequence fed into the decoder-only policy
            transformer, including the final learned action query token.
        action_logits: Mapping from action-head name to logits.
        loss: Optional summed cross-entropy over all action heads.
    """

    instruction_embedding: Tensor
    raw_visual_tokens: Tensor
    compressed_visual_tokens: Tensor
    transformer_tokens: Tensor
    action_logits: dict[str, Tensor]
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    """Root-mean-square normalization used inside the Transformer blocks."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class MultiHeadSelfAttention(nn.Module):
    """Minimal decoder-style multi-head self-attention."""

    def __init__(self, width: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")

        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, *, causal: bool = True) -> Tensor:
        """Runs self-attention over a token sequence."""

        batch_size, seq_len, _ = hidden_states.shape
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
            attention_scores = attention_scores.masked_fill(
                causal_mask,
                torch.finfo(attention_scores.dtype).min,
            )

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.width)
        return self.out_proj(context)


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


class DecoderBlock(nn.Module):
    """Pre-norm decoder block used by the RT-1 policy transformer."""

    def __init__(self, width: int, heads: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(width)
        self.attn = MultiHeadSelfAttention(width, heads, dropout=dropout)
        self.norm_2 = RMSNorm(width)
        self.ffn = FeedForward(width, mlp_ratio, dropout=dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm_1(hidden_states), causal=True)
        hidden_states = hidden_states + self.ffn(self.norm_2(hidden_states))
        return hidden_states


class InstructionEncoder(nn.Module):
    """Encodes instruction tokens into one sentence embedding.

    RT-1 uses Universal Sentence Encoder. This educational version uses
    token embeddings followed by mean pooling and a projection MLP so the
    FiLM pathway stays easy to inspect.
    """

    def __init__(self, config: RT1InstructionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.sentence_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.sentence_dim, config.sentence_dim),
        )

    def forward(self, instruction_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Returns one instruction embedding per batch element."""

        if instruction_ids.ndim != 2:
            raise ValueError(
                "instruction_ids must have shape [batch, seq_len], "
                f"got {tuple(instruction_ids.shape)}."
            )

        embeddings = self.token_embedding(instruction_ids)
        if attention_mask is None:
            attention_mask = (instruction_ids != self.config.pad_token_id).long()

        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


class FiLMModulation(nn.Module):
    """Feature-wise linear modulation with identity initialization."""

    def __init__(self, instruction_dim: int, num_channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(instruction_dim, num_channels)
        self.beta = nn.Linear(instruction_dim, num_channels)

        # RT-1 highlights that identity-initialized FiLM preserves the useful
        # behavior of the pretrained image encoder at the start of training.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feature_map: Tensor, instruction_embedding: Tensor) -> Tensor:
        """Applies FiLM modulation to a CNN feature map."""

        gamma = self.gamma(instruction_embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(instruction_embedding).unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1.0 + gamma) + beta


class FiLMedConvBlock(nn.Module):
    """MBConv-like block with FiLM modulation.

    The goal is not to reproduce EfficientNet exactly. Instead, this block keeps
    the same architectural idea: repeated convolutional stages whose activations
    are conditioned on the language instruction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        instruction_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU(approximate="tanh")
        self.film = FiLMModulation(instruction_dim, out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, feature_map: Tensor, instruction_embedding: Tensor) -> Tensor:
        """Runs the FiLM-conditioned convolution block."""

        feature_map = self.depthwise(feature_map)
        feature_map = self.pointwise(feature_map)
        feature_map = self.norm(feature_map)
        feature_map = self.film(feature_map, instruction_embedding)
        feature_map = self.act(feature_map)
        return self.dropout(feature_map)


class TokenLearner(nn.Module):
    """Compresses many visual tokens into a small learned token set."""

    def __init__(self, width: int, output_tokens: int) -> None:
        super().__init__()
        self.output_tokens = output_tokens
        self.norm = nn.LayerNorm(width)
        self.score_projection = nn.Linear(width, output_tokens)

    def forward(self, tokens: Tensor) -> Tensor:
        """Returns learned tokens of shape ``[batch, output_tokens, width]``."""

        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape [batch, num_tokens, width], got {tuple(tokens.shape)}."
            )

        normalized = self.norm(tokens)
        attention_logits = self.score_projection(normalized)
        attention_weights = attention_logits.softmax(dim=1)
        return torch.einsum("btk,btd->bkd", attention_weights, tokens)


class RT1VisionTokenizer(nn.Module):
    """Converts an image history into FiLM-conditioned visual tokens."""

    def __init__(self, config: RT1VisionConfig, instruction_dim: int) -> None:
        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.stem_width // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(config.stem_width // 2, config.stem_width, kernel_size=3, stride=2, padding=1),
            nn.GELU(approximate="tanh"),
        )
        self.blocks = nn.ModuleList(
            [
                FiLMedConvBlock(
                    config.stem_width,
                    config.stem_width,
                    stride=1,
                    instruction_dim=instruction_dim,
                    dropout=config.dropout,
                ),
                FiLMedConvBlock(
                    config.stem_width,
                    config.stem_width * 2,
                    stride=2,
                    instruction_dim=instruction_dim,
                    dropout=config.dropout,
                ),
                FiLMedConvBlock(
                    config.stem_width * 2,
                    config.stem_width * 3,
                    stride=2,
                    instruction_dim=instruction_dim,
                    dropout=config.dropout,
                ),
                FiLMedConvBlock(
                    config.stem_width * 3,
                    config.token_width,
                    stride=2,
                    instruction_dim=instruction_dim,
                    dropout=config.dropout,
                ),
            ]
        )
        self.token_learner = TokenLearner(config.token_width, config.token_learner_tokens)

    def forward(self, image_history: Tensor, instruction_embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Returns raw and TokenLearner-compressed tokens for an image history.

        Args:
            image_history: Tensor with shape ``[batch, history, channels, H, W]``.
            instruction_embedding: Tensor with shape ``[batch, instruction_dim]``.

        Returns:
            raw_tokens: ``[batch, history, 81, token_width]``
            compressed_tokens: ``[batch, history, 8, token_width]``
        """

        if image_history.ndim != 5:
            raise ValueError(
                "image_history must have shape [batch, history, channels, height, width], "
                f"got {tuple(image_history.shape)}."
            )

        batch_size, history_length, _, height, width = image_history.shape
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Expected {self.config.image_size}x{self.config.image_size} images, "
                f"got {height}x{width}."
            )
        if instruction_embedding.shape[0] != batch_size:
            raise ValueError(
                "instruction_embedding batch size must match image_history batch size, "
                f"got {instruction_embedding.shape[0]} and {batch_size}."
            )

        flattened_images = image_history.view(
            batch_size * history_length,
            image_history.shape[2],
            height,
            width,
        )
        repeated_instruction_embedding = instruction_embedding.repeat_interleave(history_length, dim=0)

        feature_map = self.stem(flattened_images)
        for block in self.blocks:
            feature_map = block(feature_map, repeated_instruction_embedding)

        feature_map = F.adaptive_avg_pool2d(
            feature_map,
            output_size=(self.config.token_grid_size, self.config.token_grid_size),
        )
        raw_tokens = feature_map.flatten(2).transpose(1, 2)
        compressed_tokens = self.token_learner(raw_tokens)

        raw_tokens = raw_tokens.view(
            batch_size,
            history_length,
            self.config.token_grid_size * self.config.token_grid_size,
            self.config.token_width,
        )
        compressed_tokens = compressed_tokens.view(
            batch_size,
            history_length,
            self.config.token_learner_tokens,
            self.config.token_width,
        )
        return raw_tokens, compressed_tokens


class RT1PolicyTransformer(nn.Module):
    """Decoder-only Transformer that predicts a non-autoregressive action state."""

    def __init__(self, config: RT1TransformerConfig, tokens_per_frame: int, token_width: int) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(token_width, config.width)
        self.action_query = nn.Parameter(torch.randn(1, 1, config.width) * 0.02)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.history_length * tokens_per_frame + 1, config.width) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    width=config.width,
                    heads=config.heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.layers)
            ]
        )
        self.final_norm = RMSNorm(config.width)

    def forward(self, compressed_visual_tokens: Tensor) -> Tensor:
        """Runs a decoder-only Transformer over the flattened visual history."""

        if compressed_visual_tokens.ndim != 4:
            raise ValueError(
                "compressed_visual_tokens must have shape [batch, history, tokens_per_frame, width], "
                f"got {tuple(compressed_visual_tokens.shape)}."
            )

        batch_size, history_length, tokens_per_frame, _ = compressed_visual_tokens.shape
        if history_length != self.config.history_length:
            raise ValueError(
                f"Expected history length {self.config.history_length}, got {history_length}."
            )

        flattened_tokens = compressed_visual_tokens.flatten(1, 2)
        hidden_states = self.input_projection(flattened_tokens)
        action_query = self.action_query.expand(batch_size, -1, -1)
        hidden_states = torch.cat([hidden_states, action_query], dim=1)
        hidden_states = hidden_states + self.position_embedding[:, : hidden_states.shape[1], :]

        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class RT1ActionHead(nn.Module):
    """Collection of discretized action heads used by RT-1."""

    def __init__(self, width: int, action_config: RT1ActionConfig) -> None:
        super().__init__()
        self.head_specs = action_config.head_specs
        self.heads = nn.ModuleDict(
            {name: nn.Linear(width, num_classes) for name, num_classes in self.head_specs}
        )

    def forward(self, hidden_states: Tensor) -> dict[str, Tensor]:
        """Returns per-head logits from the final action token."""

        return {name: head(hidden_states) for name, head in self.heads.items()}

    def loss(self, action_logits: dict[str, Tensor], action_labels: Tensor) -> Tensor:
        """Computes summed cross-entropy over all action heads.

        Args:
            action_logits: Mapping from head name to logits.
            action_labels: Tensor with shape ``[batch, num_action_heads]``.

        Returns:
            Summed cross-entropy loss over all action heads.
        """

        if action_labels.ndim != 2:
            raise ValueError(
                "action_labels must have shape [batch, num_action_heads], "
                f"got {tuple(action_labels.shape)}."
            )
        if action_labels.shape[1] != len(self.head_specs):
            raise ValueError(
                f"Expected {len(self.head_specs)} action labels, got {action_labels.shape[1]}."
            )

        total_loss = action_labels.new_zeros((), dtype=torch.float32)
        for head_index, (name, _) in enumerate(self.head_specs):
            total_loss = total_loss + F.cross_entropy(
                action_logits[name],
                action_labels[:, head_index],
                ignore_index=-100,
            )
        return total_loss


class RT1Model(nn.Module):
    """Educational RT-1 policy model."""

    def __init__(self, config: RT1Config) -> None:
        super().__init__()
        self.config = config
        self.instruction_encoder = InstructionEncoder(config.instruction_config)
        self.vision_tokenizer = RT1VisionTokenizer(
            config.vision_config,
            instruction_dim=config.instruction_config.sentence_dim,
        )
        self.policy_transformer = RT1PolicyTransformer(
            config.transformer_config,
            tokens_per_frame=config.vision_config.token_learner_tokens,
            token_width=config.vision_config.token_width,
        )
        self.action_head = RT1ActionHead(
            config.transformer_config.width,
            config.action_config,
        )

    def forward(
        self,
        *,
        image_history: Tensor,
        instruction_ids: Tensor,
        instruction_attention_mask: Tensor | None = None,
        action_labels: Tensor | None = None,
    ) -> RT1Output:
        """Runs RT-1 over an instruction and image history."""

        instruction_embedding = self.instruction_encoder(
            instruction_ids,
            attention_mask=instruction_attention_mask,
        )
        raw_visual_tokens, compressed_visual_tokens = self.vision_tokenizer(
            image_history,
            instruction_embedding,
        )
        transformer_tokens = self.policy_transformer(compressed_visual_tokens)
        final_action_state = transformer_tokens[:, -1, :]
        action_logits = self.action_head(final_action_state)

        loss = None
        if action_labels is not None:
            loss = self.action_head.loss(action_logits, action_labels)

        return RT1Output(
            instruction_embedding=instruction_embedding,
            raw_visual_tokens=raw_visual_tokens,
            compressed_visual_tokens=compressed_visual_tokens,
            transformer_tokens=transformer_tokens,
            action_logits=action_logits,
            loss=loss,
        )


def build_rt1_tiny() -> RT1Model:
    """Builds a small RT-1 variant for smoke tests and study."""

    config = RT1Config(
        instruction_config=RT1InstructionConfig(
            vocab_size=256,
            embed_dim=64,
            sentence_dim=96,
            pad_token_id=0,
        ),
        vision_config=RT1VisionConfig(
            image_size=48,
            in_channels=3,
            stem_width=32,
            token_width=96,
            token_grid_size=9,
            token_learner_tokens=4,
            dropout=0.0,
        ),
        transformer_config=RT1TransformerConfig(
            width=128,
            layers=4,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            history_length=6,
        ),
        action_config=RT1ActionConfig(
            continuous_bins=32,
            mode_classes=3,
        ),
    )
    return RT1Model(config)


def _sample_tiny_action_labels(model: RT1Model, batch_size: int) -> Tensor:
    """Creates random action labels matching the tiny action-head sizes."""

    labels = []
    for _, num_classes in model.action_head.head_specs:
        labels.append(torch.randint(0, num_classes, (batch_size,), dtype=torch.long))
    return torch.stack(labels, dim=1)


def _smoke_test() -> None:
    """Runs a small RT-1 forward pass and prints key shapes."""

    model = build_rt1_tiny()
    batch_size = 2
    image_history = torch.randn(
        batch_size,
        model.config.transformer_config.history_length,
        3,
        model.config.vision_config.image_size,
        model.config.vision_config.image_size,
    )
    instruction_ids = torch.tensor(
        [
            [11, 21, 31, 41, 0, 0],
            [12, 22, 32, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    instruction_attention_mask = (instruction_ids != 0).long()
    action_labels = _sample_tiny_action_labels(model, batch_size)

    output = model(
        image_history=image_history,
        instruction_ids=instruction_ids,
        instruction_attention_mask=instruction_attention_mask,
        action_labels=action_labels,
    )
    print("instruction_embedding:", tuple(output.instruction_embedding.shape))
    print("raw_visual_tokens:", tuple(output.raw_visual_tokens.shape))
    print("compressed_visual_tokens:", tuple(output.compressed_visual_tokens.shape))
    print("transformer_tokens:", tuple(output.transformer_tokens.shape))
    print(
        "action_logits:",
        {name: tuple(logits.shape) for name, logits in output.action_logits.items()},
    )
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _smoke_test()
