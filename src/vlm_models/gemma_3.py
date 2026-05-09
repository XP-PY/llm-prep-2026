"""A compact educational Gemma 3 implementation in PyTorch.

This module mirrors the architecture summary in ``docs/Large_Models/Gemma_3.md``:

1. A frozen SigLIP-like vision tower converts each image crop into patch tokens.
2. A learned soft-token compressor condenses each crop into a fixed set of 256
   visual tokens before the language model sees them.
3. A decoder-only language model uses Grouped-Query Attention (GQA),
   QK-normalization, and 5:1 local/global attention interleaving.
4. Pan & Scan is modeled as an inference-side preprocessing step that turns a
   non-square image into a grid of non-overlapping square crops.

This is an educational implementation rather than a checkpoint-faithful replica:

* The vision tower is a ViT-style encoder that plays the role of SigLIP.
* The 256 soft tokens are produced by learned cross-attention queries instead of
  reproducing Google's exact visual adapter.
* RoPE is implemented in a simplified form without the full interpolation recipe
  used by production long-context models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from .clip import MultiHeadSelfAttention, VisionTransformerConfig
    from .deepseek_vl import PatchTokenVisionTransformer
except ImportError:  # pragma: no cover - allows direct file execution for smoke tests
    from clip import MultiHeadSelfAttention, VisionTransformerConfig
    from deepseek_vl import PatchTokenVisionTransformer


@dataclass(frozen=True)
class Gemma3VisionConfig:
    """Configuration for Gemma 3 vision preprocessing and encoding.

    Attributes:
        image_size: Target square crop size for the SigLIP-like vision tower.
        max_crops: Maximum number of Pan & Scan crops per image.
        encoder_config: ViT-style configuration used for the frozen vision tower.
        soft_tokens_per_crop: Number of compressed visual tokens per crop.
        compressor_heads: Attention heads used by the soft-token compressor.
    """

    image_size: int = 896
    max_crops: int = 4
    encoder_config: VisionTransformerConfig = field(
        default_factory=lambda: VisionTransformerConfig(
            image_size=896,
            patch_size=16,
            in_channels=3,
            width=256,
            layers=6,
            heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
        )
    )
    soft_tokens_per_crop: int = 256
    compressor_heads: int = 8


@dataclass(frozen=True)
class Gemma3LanguageConfig:
    """Configuration for the Gemma 3 decoder-only language model.

    Attributes:
        vocab_size: Vocabulary size of the language model.
        context_length: Maximum number of text tokens.
        width: Hidden width of the decoder.
        layers: Number of decoder blocks.
        num_query_heads: Number of query heads in GQA.
        num_kv_heads: Number of key/value heads in GQA.
        mlp_ratio: Expansion ratio of the feed-forward block.
        dropout: Dropout used inside attention and MLP blocks.
        pad_token_id: Padding token ID used for optional masking.
        local_window_size: Window span for local sliding-window attention.
        local_to_global_ratio: Number of local layers before each global layer.
        local_rope_base: RoPE base for local layers.
        global_rope_base: RoPE base for global layers.
    """

    vocab_size: int = 262_144
    context_length: int = 128_000
    width: int = 2048
    layers: int = 18
    num_query_heads: int = 16
    num_kv_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0
    local_window_size: int = 1024
    local_to_global_ratio: int = 5
    local_rope_base: float = 10_000.0
    global_rope_base: float = 1_000_000.0


@dataclass(frozen=True)
class Gemma3Config:
    """Top-level configuration for the educational Gemma 3 model."""

    vision_config: Gemma3VisionConfig = field(default_factory=Gemma3VisionConfig)
    language_config: Gemma3LanguageConfig = field(default_factory=Gemma3LanguageConfig)


@dataclass
class Gemma3Output:
    """Output container returned by ``Gemma3Model.forward``.

    Attributes:
        visual_tokens: Padded visual token tensor of shape
            ``[batch, max_visual_len, hidden]``.
        visual_attention_mask: Mask indicating which visual tokens are valid.
        logits: Text next-token logits.
        loss: Optional causal language modeling loss.
        chosen_pan_scan_grids: Crop grids chosen for each image.
        num_crops_per_image: Number of visual crops used for each image.
    """

    visual_tokens: Tensor
    visual_attention_mask: Tensor
    logits: Tensor
    loss: Tensor | None
    chosen_pan_scan_grids: list[tuple[int, int]]
    num_crops_per_image: list[int]


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
    """Builds cosine and sine caches for RoPE."""

    half_dim = head_dim // 2
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    frequency_exponents = torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
    inverse_frequencies = base ** (-frequency_exponents)
    angles = torch.outer(positions, inverse_frequencies)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1).to(dtype)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def apply_rope(hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Applies rotary positional embedding to query or key states.

    Args:
        hidden_states: Tensor with shape ``[batch, num_heads, seq_len, head_dim]``.
        cos: Cosine cache with shape ``[seq_len, head_dim]``.
        sin: Sine cache with shape ``[seq_len, head_dim]``.

    Returns:
        RoPE-transformed hidden states with the same shape as the input.
    """

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return hidden_states * cos + rotate_half(hidden_states) * sin


class SoftTokenCompressor(nn.Module):
    """Condenses patch tokens into a fixed number of soft visual tokens.

    Gemma 3 feeds 256 soft tokens from the vision tower into the language model.
    This module uses learned query tokens and cross-attention to implement that
    compression in a compact and readable way.
    """

    def __init__(self, vision_width: int, llm_width: int, num_soft_tokens: int, heads: int) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_soft_tokens, vision_width) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=heads,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(vision_width)
        self.projection = nn.Sequential(
            nn.Linear(vision_width, llm_width),
            nn.GELU(),
            nn.Linear(llm_width, llm_width),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """Returns fixed visual tokens of shape ``[batch, num_soft_tokens, llm_width]``."""

        queries = self.query_tokens.expand(patch_tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(queries, patch_tokens, patch_tokens, need_weights=False)
        return self.projection(self.ln(attended))


class PanAndScanVisionEncoder(nn.Module):
    """Vision encoder with Pan & Scan preprocessing.

    The main idea is:

    * square images -> one 896x896 crop
    * wide or tall images -> resized/padded canvas -> split into non-overlapping
      square crops -> each crop becomes 256 soft tokens
    """

    def __init__(self, config: Gemma3VisionConfig, llm_width: int) -> None:
        super().__init__()
        self.config = config
        self.vision_tower = PatchTokenVisionTransformer(config.encoder_config)
        self.soft_token_compressor = SoftTokenCompressor(
            vision_width=config.encoder_config.width,
            llm_width=llm_width,
            num_soft_tokens=config.soft_tokens_per_crop,
            heads=config.compressor_heads,
        )

        # Gemma 3 freezes the vision encoder in released multimodal variants.
        for parameter in self.vision_tower.parameters():
            parameter.requires_grad = False

    def _choose_crop_grid(self, image_height: int, image_width: int) -> tuple[int, int]:
        """Chooses a Pan & Scan crop grid under the configured crop budget."""

        aspect_ratio = image_width / max(image_height, 1)
        best_grid = (1, 1)
        best_score = float("inf")

        for rows in range(1, self.config.max_crops + 1):
            for cols in range(1, self.config.max_crops + 1):
                if rows * cols > self.config.max_crops:
                    continue
                grid_ratio = cols / rows
                padding_score = abs(aspect_ratio - grid_ratio)
                area_penalty = 0.01 * (rows * cols - 1)
                score = padding_score + area_penalty
                if score < best_score:
                    best_score = score
                    best_grid = (rows, cols)

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

    def _split_into_crops(self, image: Tensor, rows: int, cols: int) -> Tensor:
        """Splits an image into non-overlapping square crops."""

        tile_size = self.config.image_size
        canvas = self._resize_with_padding(
            image,
            target_height=rows * tile_size,
            target_width=cols * tile_size,
        )

        crops = []
        for row in range(rows):
            for col in range(cols):
                top = row * tile_size
                left = col * tile_size
                crop = canvas[:, top : top + tile_size, left : left + tile_size]
                crops.append(crop)
        return torch.stack(crops, dim=0)

    def encode_images(
        self,
        pixel_values: Tensor | list[Tensor],
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]], list[int]]:
        """Encodes one or more images into padded visual token sequences."""

        if isinstance(pixel_values, Tensor):
            if pixel_values.ndim != 4:
                raise ValueError(
                    "pixel_values tensor must have shape [batch, channels, height, width], "
                    f"got {tuple(pixel_values.shape)}."
                )
            images = [image for image in pixel_values]
        else:
            images = list(pixel_values)
            if not images:
                raise ValueError("pixel_values list cannot be empty.")

        visual_sequences: list[Tensor] = []
        chosen_grids: list[tuple[int, int]] = []
        num_crops_per_image: list[int] = []

        for image in images:
            if image.ndim != 3:
                raise ValueError(
                    f"Each image must have shape [channels, height, width], got {tuple(image.shape)}."
                )

            _, image_height, image_width = image.shape
            rows, cols = self._choose_crop_grid(image_height, image_width)
            crops = self._split_into_crops(image, rows, cols)

            with torch.no_grad():
                patch_tokens = self.vision_tower(crops)
            crop_soft_tokens = self.soft_token_compressor(patch_tokens)
            image_sequence = crop_soft_tokens.reshape(-1, crop_soft_tokens.shape[-1])

            visual_sequences.append(image_sequence)
            chosen_grids.append((rows, cols))
            num_crops_per_image.append(rows * cols)

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

        return padded_visual_tokens, visual_attention_mask, chosen_grids, num_crops_per_image


class Gemma3GQAAttention(nn.Module):
    """Grouped-Query Attention with QK-normalization and local/global modes."""

    def __init__(
        self,
        width: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width % num_query_heads != 0:
            raise ValueError("width must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")

        self.width = width
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = width // num_query_heads
        self.kv_repeat = num_query_heads // num_kv_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _build_attention_mask(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        local_window_size: int | None,
    ) -> Tensor:
        """Builds a causal attention mask with optional sliding window."""

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
        """Runs GQA in either local or global attention mode."""

        batch_size, sequence_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
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

        # QK-Norm keeps attention logits stable by normalizing query/key vectors
        # before the dot product, which is especially useful in long-context models.
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        key = key.repeat_interleave(self.kv_repeat, dim=1)
        value = value.repeat_interleave(self.kv_repeat, dim=1)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        structural_mask = self._build_attention_mask(
            sequence_length,
            device=hidden_states.device,
            local_window_size=local_window_size,
        )
        attention_scores = attention_scores.masked_fill(
            structural_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(attention_scores.dtype).min,
        )

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            attention_scores = attention_scores.masked_fill(
                ~key_mask,
                torch.finfo(attention_scores.dtype).min,
            )

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.width)
        return self.out_proj(context)


class Gemma3FeedForward(nn.Module):
    """A compact feed-forward block for Gemma 3 decoder layers."""

    def __init__(self, width: int, mlp_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return self.dropout(hidden_states)


class Gemma3DecoderBlock(nn.Module):
    """Decoder block with GQA and local/global attention selection."""

    def __init__(self, config: Gemma3LanguageConfig, *, is_global: bool) -> None:
        super().__init__()
        self.is_global = is_global
        self.local_window_size = None if is_global else config.local_window_size
        self.rope_base = config.global_rope_base if is_global else config.local_rope_base
        self.norm_1 = RMSNorm(config.width)
        self.attn = Gemma3GQAAttention(
            width=config.width,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
        )
        self.norm_2 = RMSNorm(config.width)
        self.ffn = Gemma3FeedForward(
            width=config.width,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm_1(hidden_states),
            attention_mask=attention_mask,
            local_window_size=self.local_window_size,
            rope_base=self.rope_base,
        )
        hidden_states = hidden_states + self.ffn(self.norm_2(hidden_states))
        return hidden_states


class Gemma3LanguageModel(nn.Module):
    """Decoder-only Gemma 3 backbone with 5:1 local/global interleaving."""

    def __init__(self, config: Gemma3LanguageConfig, max_visual_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.max_visual_tokens = max_visual_tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList()
        cycle_length = config.local_to_global_ratio + 1
        for layer_index in range(config.layers):
            is_global = (layer_index % cycle_length) == (cycle_length - 1)
            self.blocks.append(Gemma3DecoderBlock(config, is_global=is_global))
        self.final_norm = RMSNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        visual_tokens: Tensor,
        visual_attention_mask: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Runs causal decoding over visual-prefix and text tokens."""

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, text_len], got {tuple(input_ids.shape)}."
            )
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"visual_tokens must have shape [batch, visual_len, hidden], got {tuple(visual_tokens.shape)}."
            )
        if visual_attention_mask.ndim != 2:
            raise ValueError(
                "visual_attention_mask must have shape [batch, visual_len], "
                f"got {tuple(visual_attention_mask.shape)}."
            )

        batch_size, text_len = input_ids.shape
        _, visual_len, hidden_width = visual_tokens.shape
        if hidden_width != self.config.width:
            raise ValueError(
                f"visual token width {hidden_width} does not match language width {self.config.width}."
            )
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        text_embeddings = self.token_embedding(input_ids)
        hidden_states = torch.cat([visual_tokens, text_embeddings], dim=1)
        full_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=full_attention_mask)

        hidden_states = self.final_norm(hidden_states)
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


class Gemma3Model(nn.Module):
    """Educational implementation of Gemma 3 multimodal decoding."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = PanAndScanVisionEncoder(
            config.vision_config,
            llm_width=config.language_config.width,
        )
        max_visual_tokens = config.vision_config.soft_tokens_per_crop * config.vision_config.max_crops
        self.language_model = Gemma3LanguageModel(
            config.language_config,
            max_visual_tokens=max_visual_tokens,
        )

    def encode_images(
        self,
        pixel_values: Tensor | list[Tensor],
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]], list[int]]:
        """Encodes images into padded visual tokens."""

        return self.vision_encoder.encode_images(pixel_values)

    def forward(
        self,
        pixel_values: Tensor | list[Tensor],
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Gemma3Output:
        """Runs multimodal Gemma 3 decoding."""

        visual_tokens, visual_attention_mask, chosen_grids, num_crops_per_image = self.encode_images(
            pixel_values
        )
        logits, loss = self.language_model(
            input_ids=input_ids,
            visual_tokens=visual_tokens,
            visual_attention_mask=visual_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
        )
        return Gemma3Output(
            visual_tokens=visual_tokens,
            visual_attention_mask=visual_attention_mask,
            logits=logits,
            loss=loss,
            chosen_pan_scan_grids=chosen_grids,
            num_crops_per_image=num_crops_per_image,
        )


def build_gemma_3_tiny() -> Gemma3Model:
    """Builds a small Gemma 3 variant for smoke tests and study."""

    config = Gemma3Config(
        vision_config=Gemma3VisionConfig(
            image_size=32,
            max_crops=4,
            encoder_config=VisionTransformerConfig(
                image_size=32,
                patch_size=8,
                in_channels=3,
                width=96,
                layers=2,
                heads=4,
                mlp_ratio=4.0,
                dropout=0.0,
            ),
            soft_tokens_per_crop=16,
            compressor_heads=4,
        ),
        language_config=Gemma3LanguageConfig(
            vocab_size=512,
            context_length=32,
            width=128,
            layers=6,
            num_query_heads=4,
            num_kv_heads=2,
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
            local_window_size=8,
            local_to_global_ratio=5,
            local_rope_base=10_000.0,
            global_rope_base=1_000_000.0,
        ),
    )
    return Gemma3Model(config)


def _smoke_test() -> None:
    """Runs a small forward pass to verify tensor shapes."""

    model = build_gemma_3_tiny()
    images = [
        torch.randn(3, 32, 48),
        torch.randn(3, 48, 32),
    ]
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
    print("visual_tokens:", tuple(output.visual_tokens.shape))
    print("visual_attention_mask:", tuple(output.visual_attention_mask.shape))
    print("logits:", tuple(output.logits.shape))
    print("chosen_pan_scan_grids:", output.chosen_pan_scan_grids)
    print("num_crops_per_image:", output.num_crops_per_image)
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _smoke_test()
