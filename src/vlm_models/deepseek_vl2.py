"""A compact educational DeepSeek-VL2 implementation in PyTorch.

This module follows the high-level architecture described in
``docs/Large_Models/DeepSeek_VL2.md``:

1. A single SigLIP-like vision tower is applied to a global thumbnail and a
   grid of local tiles chosen by dynamic tiling.
2. A pixel-shuffle-style compressor reduces each tile from a dense feature grid
   to a smaller token budget while preserving local neighborhood information.
3. Learned separator tokens structure the visual sequence before it is fed into
   a decoder-only language model.
4. The language backbone uses a simplified MoE feed-forward stack to mirror the
   sparse routing theme of DeepSeek-VL2.

This implementation is intentionally educational rather than checkpoint-faithful:

* The vision tower is a ViT-like patch encoder followed by feature-map resizing,
  not the exact released SigLIP-SO400M checkpoint.
* The decoder uses standard multi-head attention instead of true MLA.
* The MoE stack is dense-in-code but sparse-in-weighting: it computes all expert
  outputs for readability and then mixes the selected ones according to routing.
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
class DeepSeekVL2VisionConfig:
    """Configuration for the dynamic-tiling vision pathway.

    Attributes:
        tile_size: Resolution of each local tile and the global thumbnail.
        max_tiles: Maximum number of local tiles allowed for one image.
        disable_tiling_above_n_images: If a batch contains more than this many
            images, dynamic tiling is disabled and the model falls back to a
            single global view per image to cap the token budget.
        encoder_config: SigLIP-like patch encoder configuration.
        raw_token_grid_size: Grid size after resizing tile features. The paper
            uses 27x27 = 729 tokens before pixel-shuffle compression.
        compressed_token_grid_size: Token grid size after pixel-shuffle-style
            compression. The paper uses 14x14 = 196 tokens.
    """

    tile_size: int = 384
    max_tiles: int = 9
    disable_tiling_above_n_images: int = 2
    encoder_config: VisionTransformerConfig = field(
        default_factory=lambda: VisionTransformerConfig(
            image_size=384,
            patch_size=16,
            in_channels=3,
            width=256,
            layers=6,
            heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
        )
    )
    raw_token_grid_size: int = 27
    compressed_token_grid_size: int = 14


@dataclass(frozen=True)
class DeepSeekVL2LanguageConfig:
    """Configuration for the simplified MoE language model.

    Attributes:
        vocab_size: Vocabulary size of the decoder.
        context_length: Maximum number of text tokens.
        width: Hidden width of the decoder.
        layers: Number of decoder blocks.
        heads: Number of attention heads.
        mlp_ratio: Expansion ratio for each expert MLP.
        dropout: Dropout used in attention and expert MLPs.
        pad_token_id: Padding token ID for optional text masking.
        num_routed_experts: Number of routed experts in each MoE block.
        num_shared_experts: Number of shared experts evaluated on every token.
        top_k_experts: Number of routed experts selected per token.
        routing_function: Either ``"softmax"`` or ``"sigmoid"``.
    """

    vocab_size: int = 32_000
    context_length: int = 256
    width: int = 1024
    layers: int = 12
    heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    pad_token_id: int = 0
    num_routed_experts: int = 8
    num_shared_experts: int = 2
    top_k_experts: int = 2
    routing_function: str = "softmax"


@dataclass(frozen=True)
class DeepSeekVL2Config:
    """Top-level configuration for the educational DeepSeek-VL2 model."""

    vision_config: DeepSeekVL2VisionConfig = field(default_factory=DeepSeekVL2VisionConfig)
    language_config: DeepSeekVL2LanguageConfig = field(default_factory=DeepSeekVL2LanguageConfig)
    adapter_hidden_dim: int = 1024


@dataclass
class DeepSeekVL2Output:
    """Output container returned by ``DeepSeekVL2Model.forward``.

    Attributes:
        visual_tokens: Padded visual token tensor after tiling, compression, and
            special-token insertion.
        visual_attention_mask: Mask indicating which visual tokens are valid.
        logits: Text next-token logits with shape ``[batch, text_len, vocab_size]``.
        loss: Optional causal language modeling loss.
        chosen_tile_grids: The ``(rows, cols)`` grid selected for each image.
        num_tiles_per_image: Number of local tiles chosen for each image.
    """

    visual_tokens: Tensor
    visual_attention_mask: Tensor
    logits: Tensor
    loss: Tensor | None
    chosen_tile_grids: list[tuple[int, int]]
    num_tiles_per_image: list[int]


class FeedForwardExpert(nn.Module):
    """One expert MLP inside the simplified MoE block."""

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


class MoEFeedForward(nn.Module):
    """A readable MoE feed-forward layer with routed and shared experts."""

    def __init__(
        self,
        width: int,
        mlp_ratio: float,
        num_routed_experts: int,
        num_shared_experts: int,
        top_k_experts: int,
        routing_function: str = "softmax",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k_experts > num_routed_experts:
            raise ValueError("top_k_experts cannot exceed num_routed_experts.")
        if routing_function not in {"softmax", "sigmoid"}:
            raise ValueError("routing_function must be 'softmax' or 'sigmoid'.")

        self.num_routed_experts = num_routed_experts
        self.top_k_experts = top_k_experts
        self.routing_function = routing_function

        self.router = nn.Linear(width, num_routed_experts)
        self.routed_experts = nn.ModuleList(
            [FeedForwardExpert(width, mlp_ratio, dropout=dropout) for _ in range(num_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [FeedForwardExpert(width, mlp_ratio, dropout=dropout) for _ in range(num_shared_experts)]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Routes tokens through experts and returns the mixed output."""

        router_logits = self.router(hidden_states)
        if self.routing_function == "softmax":
            routing_scores = router_logits.softmax(dim=-1)
        else:
            routing_scores = router_logits.sigmoid()

        topk_scores, topk_indices = torch.topk(
            routing_scores,
            k=self.top_k_experts,
            dim=-1,
        )
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.routed_experts],
            dim=2,
        )
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_states.shape[-1])
        selected_outputs = torch.gather(expert_outputs, dim=2, index=gather_index)
        mixed_routed = (selected_outputs * topk_scores.unsqueeze(-1)).sum(dim=2)

        if self.shared_experts:
            shared_outputs = torch.stack(
                [expert(hidden_states) for expert in self.shared_experts],
                dim=0,
            ).mean(dim=0)
        else:
            shared_outputs = 0.0

        return mixed_routed + shared_outputs


class DeepSeekVL2DecoderBlock(nn.Module):
    """Decoder block with causal self-attention and MoE feed-forward."""

    def __init__(self, config: DeepSeekVL2LanguageConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.width)
        self.attn = MultiHeadSelfAttention(config.width, config.heads, dropout=config.dropout)
        self.ln_2 = nn.LayerNorm(config.width)
        self.moe = MoEFeedForward(
            width=config.width,
            mlp_ratio=config.mlp_ratio,
            num_routed_experts=config.num_routed_experts,
            num_shared_experts=config.num_shared_experts,
            top_k_experts=config.top_k_experts,
            routing_function=config.routing_function,
            dropout=config.dropout,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.ln_1(hidden_states),
            causal=True,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.moe(self.ln_2(hidden_states))
        return hidden_states


class DeepSeekVL2LanguageModel(nn.Module):
    """Decoder-only language model with simplified MoE blocks.

    The official model uses MLA for KV compression. This implementation keeps
    the decoder interface but uses standard causal self-attention for clarity.
    """

    def __init__(self, config: DeepSeekVL2LanguageConfig, max_visual_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.max_visual_tokens = max_visual_tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_visual_tokens + config.context_length, config.width) * 0.01
        )
        self.blocks = nn.ModuleList([DeepSeekVL2DecoderBlock(config) for _ in range(config.layers)])
        self.ln_final = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        visual_tokens: Tensor,
        visual_attention_mask: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Runs autoregressive decoding over visual tokens plus text tokens."""

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
        hidden_states = hidden_states + self.position_embedding[:, : visual_len + text_len, :]
        full_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=full_attention_mask)

        hidden_states = self.ln_final(hidden_states)
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


class TileFeatureProjector(nn.Module):
    """Processes one 384x384 view into a 27x27 feature grid.

    The paper uses SigLIP-SO400M to produce 729 embeddings per tile. Here we
    approximate that behavior by reusing a ViT-like patch encoder and resizing
    its token grid to 27x27.
    """

    def __init__(self, encoder_config: VisionTransformerConfig, raw_token_grid_size: int) -> None:
        super().__init__()
        self.encoder = PatchTokenVisionTransformer(encoder_config)
        self.raw_token_grid_size = raw_token_grid_size
        self.encoder_grid_size = int(encoder_config.num_patches**0.5)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Returns tile features of shape ``[batch, 27, 27, hidden]``."""

        patch_tokens = self.encoder(pixel_values)
        batch_size, _, hidden = patch_tokens.shape
        feature_grid = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            hidden,
            self.encoder_grid_size,
            self.encoder_grid_size,
        )
        feature_grid = F.interpolate(
            feature_grid,
            size=(self.raw_token_grid_size, self.raw_token_grid_size),
            mode="bilinear",
            align_corners=False,
        )
        return feature_grid.permute(0, 2, 3, 1)


class PixelShuffleTokenCompressor(nn.Module):
    """Compresses a 27x27 tile grid into a 14x14 token grid.

    The paper describes a 2x2 pixel-shuffle style compression from 729 tokens to
    196 tokens. We mimic that by padding to an even grid, grouping 2x2 spatial
    neighborhoods, and projecting the concatenated channel features back down.
    """

    def __init__(self, input_width: int, output_width: int, output_grid_size: int) -> None:
        super().__init__()
        self.output_grid_size = output_grid_size
        self.projection = nn.Linear(input_width * 4, output_width)

    def forward(self, tile_grid: Tensor) -> Tensor:
        """Compresses tile features from ``[B, 27, 27, D]`` to ``[B, 14, 14, D_out]``."""

        batch_size, height, width, channels = tile_grid.shape
        if height % 2 != 0 or width % 2 != 0:
            padded_height = height + (height % 2)
            padded_width = width + (width % 2)
            padded_grid = torch.zeros(
                batch_size,
                padded_height,
                padded_width,
                channels,
                device=tile_grid.device,
                dtype=tile_grid.dtype,
            )
            padded_grid[:, :height, :width, :] = tile_grid
            tile_grid = padded_grid
            height, width = padded_height, padded_width

        tile_grid = tile_grid.reshape(batch_size, height // 2, 2, width // 2, 2, channels)
        tile_grid = tile_grid.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size,
            height // 2,
            width // 2,
            channels * 4,
        )
        compressed = self.projection(tile_grid)
        if compressed.shape[1] != self.output_grid_size or compressed.shape[2] != self.output_grid_size:
            compressed = compressed.permute(0, 3, 1, 2)
            compressed = F.interpolate(
                compressed,
                size=(self.output_grid_size, self.output_grid_size),
                mode="bilinear",
                align_corners=False,
            )
            compressed = compressed.permute(0, 2, 3, 1)
        return compressed


class DynamicTilingVisionEncoder(nn.Module):
    """Single-tower dynamic-tiling vision encoder.

    For each image, this module chooses a local tile grid that best matches the
    image aspect ratio while keeping the total tile count under a configured
    budget. It also creates a global thumbnail, compresses all tile features,
    inserts learned structural tokens, and pads the resulting visual sequences.
    """

    def __init__(self, config: DeepSeekVL2VisionConfig, llm_width: int, adapter_hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self.tile_encoder = TileFeatureProjector(
            config.encoder_config,
            raw_token_grid_size=config.raw_token_grid_size,
        )
        self.tile_compressor = PixelShuffleTokenCompressor(
            input_width=config.encoder_config.width,
            output_width=adapter_hidden_dim,
            output_grid_size=config.compressed_token_grid_size,
        )
        self.adapter = nn.Sequential(
            nn.Linear(adapter_hidden_dim, llm_width),
            nn.GELU(),
            nn.Linear(llm_width, llm_width),
        )
        self.tile_newline = nn.Parameter(torch.randn(1, 1, llm_width) * 0.02)
        self.view_separator = nn.Parameter(torch.randn(1, 1, llm_width) * 0.02)

    def _choose_tile_grid(self, image_height: int, image_width: int, allow_tiling: bool) -> tuple[int, int]:
        """Chooses the local tile grid with minimum aspect mismatch."""

        if not allow_tiling:
            return 1, 1

        aspect_ratio = image_width / max(image_height, 1)
        best_grid = (1, 1)
        best_score = float("inf")

        for rows in range(1, self.config.max_tiles + 1):
            for cols in range(1, self.config.max_tiles + 1):
                if rows * cols > self.config.max_tiles:
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
        """Resizes an image to fit inside a canvas and pads the remainder."""

        # Input image shape: [C, H, W].
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
        # Resized image shape: [C, resized_height, resized_width].

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
        # Letterboxed canvas shape: [C, target_height, target_width].
        return canvas

    def _split_local_tiles(self, image: Tensor, rows: int, cols: int) -> Tensor:
        """Creates local tiles by letterboxing to the chosen canvas and splitting."""

        # Canvas shape: [C, rows * tile_size, cols * tile_size].
        canvas = self._resize_with_padding(
            image,
            target_height=rows * self.config.tile_size,
            target_width=cols * self.config.tile_size,
        )
        tiles = []
        for row in range(rows):
            for col in range(cols):
                top = row * self.config.tile_size
                left = col * self.config.tile_size
                # Each tile shape: [C, tile_size, tile_size].
                tile = canvas[:, top : top + self.config.tile_size, left : left + self.config.tile_size]
                tiles.append(tile)
        # Stacked local tiles shape: [rows * cols, C, tile_size, tile_size].
        return torch.stack(tiles, dim=0)

    def _grid_to_sequence_with_newlines(self, tile_grid: Tensor) -> Tensor:
        """Converts a 14x14 tile grid into 210 tokens with row separators."""

        # Input tile grid shape: [grid_h, grid_w, D], e.g. [14, 14, D].
        rows = []
        for row_tokens in tile_grid:
            # Row token shape: [grid_w, D], newline token shape: [1, D].
            rows.append(row_tokens)
            rows.append(self.tile_newline.expand(1, -1, -1).squeeze(0))
        # Output sequence shape: [grid_h * (grid_w + 1), D], e.g. [14 * 15, D] = [210, D].
        return torch.cat(rows, dim=0)

    def encode_images(
        self,
        pixel_values: Tensor | list[Tensor],
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]], list[int]]:
        """Encodes one or more images into padded visual token sequences.

        Args:
            pixel_values: Either a batch tensor ``[B, C, H, W]`` or a Python list
                of image tensors ``[C, H, W]`` with variable shapes.

        Returns:
            Tuple ``(visual_tokens, visual_mask, chosen_grids, num_tiles_per_image)``.
        """

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

        allow_tiling = len(images) <= self.config.disable_tiling_above_n_images

        visual_sequences: list[Tensor] = []
        chosen_grids: list[tuple[int, int]] = []
        num_tiles_per_image: list[int] = []

        for image in images:
            if image.ndim != 3:
                raise ValueError(
                    f"Each image must have shape [channels, height, width], got {tuple(image.shape)}."
                )

            # Original image shape: [C, H, W].
            _, image_height, image_width = image.shape
            rows, cols = self._choose_tile_grid(image_height, image_width, allow_tiling=allow_tiling)
            # Local tiles shape after splitting: [num_local_tiles, C, tile_size, tile_size].
            local_tiles = self._split_local_tiles(image, rows, cols)
            global_thumbnail = F.interpolate(
                image.unsqueeze(0),
                size=(self.config.tile_size, self.config.tile_size),
                mode="bilinear",
                align_corners=False,
            )
            # Global thumbnail shape: [1, C, tile_size, tile_size].

            # All views shape: [1 + num_local_tiles, C, tile_size, tile_size].
            all_views = torch.cat([global_thumbnail, local_tiles], dim=0)
            # Raw features shape: [1 + num_local_tiles, raw_grid, raw_grid, vision_width].
            raw_features = self.tile_encoder(all_views)
            # Compressed features shape: [1 + num_local_tiles, compressed_grid, compressed_grid, adapter_hidden_dim].
            compressed_features = self.tile_compressor(raw_features)
            # Flatten one tile/grid into a token list:
            # [num_views, compressed_grid * compressed_grid, adapter_hidden_dim].
            adapted_tokens = self.adapter(compressed_features.reshape(
                compressed_features.shape[0],
                -1,
                compressed_features.shape[-1],
            ))
            # Restore per-tile 2D layout:
            # [num_views, compressed_grid, compressed_grid, llm_width].
            adapted_tokens = adapted_tokens.reshape(
                compressed_features.shape[0],
                self.config.compressed_token_grid_size,
                self.config.compressed_token_grid_size,
                -1,
            )

            # Global sequence shape: [compressed_grid * (compressed_grid + 1), llm_width].
            global_sequence = self._grid_to_sequence_with_newlines(adapted_tokens[0])
            # Each local sequence has the same shape as the global sequence.
            local_sequences = [self._grid_to_sequence_with_newlines(tile_grid) for tile_grid in adapted_tokens[1:]]
            if local_sequences:
                # Final image sequence shape:
                # [global_tokens + 1 separator + num_local_tiles * local_tokens, llm_width].
                image_sequence = torch.cat(
                    [global_sequence, self.view_separator.expand(1, -1, -1).squeeze(0)] + local_sequences,
                    dim=0,
                )
            else:
                image_sequence = global_sequence

            visual_sequences.append(image_sequence)
            chosen_grids.append((rows, cols))
            num_tiles_per_image.append(rows * cols)

        # Pad variable-length visual sequences into one batch tensor:
        # [batch, max_visual_len, llm_width].
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

        # Attention mask shape: [batch, max_visual_len].
        return padded_visual_tokens, visual_attention_mask, chosen_grids, num_tiles_per_image


class DeepSeekVL2Model(nn.Module):
    """Educational implementation of DeepSeek-VL2."""

    def __init__(self, config: DeepSeekVL2Config) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = DynamicTilingVisionEncoder(
            config.vision_config,
            llm_width=config.language_config.width,
            adapter_hidden_dim=config.adapter_hidden_dim,
        )

        # Worst-case visual length:
        # one global tile + max local tiles, each becoming 14 rows * (14 tokens + 1 newline)
        # plus one view separator between global and local tiles.
        tokens_per_view = config.vision_config.compressed_token_grid_size * (
            config.vision_config.compressed_token_grid_size + 1
        )
        max_views = config.vision_config.max_tiles + 1
        max_visual_tokens = tokens_per_view * max_views + 1
        self.language_model = DeepSeekVL2LanguageModel(
            config.language_config,
            max_visual_tokens=max_visual_tokens,
        )

    def encode_images(
        self,
        pixel_values: Tensor | list[Tensor],
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]], list[int]]:
        """Encodes images into padded visual token sequences."""

        return self.visual_encoder.encode_images(pixel_values)

    def forward(
        self,
        pixel_values: Tensor | list[Tensor],
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> DeepSeekVL2Output:
        """Runs multimodal decoding with dynamic-tiling vision inputs."""

        visual_tokens, visual_attention_mask, chosen_grids, num_tiles_per_image = self.encode_images(
            pixel_values
        )
        logits, loss = self.language_model(
            input_ids=input_ids,
            visual_tokens=visual_tokens,
            visual_attention_mask=visual_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
        )
        return DeepSeekVL2Output(
            visual_tokens=visual_tokens,
            visual_attention_mask=visual_attention_mask,
            logits=logits,
            loss=loss,
            chosen_tile_grids=chosen_grids,
            num_tiles_per_image=num_tiles_per_image,
        )


def build_deepseek_vl2_tiny() -> DeepSeekVL2Model:
    """Builds a tiny DeepSeek-VL2 variant for smoke tests and study."""

    config = DeepSeekVL2Config(
        vision_config=DeepSeekVL2VisionConfig(
            tile_size=32,
            max_tiles=4,
            disable_tiling_above_n_images=2,
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
            raw_token_grid_size=7,
            compressed_token_grid_size=4,
        ),
        language_config=DeepSeekVL2LanguageConfig(
            vocab_size=512,
            context_length=32,
            width=128,
            layers=2,
            heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            pad_token_id=0,
            num_routed_experts=4,
            num_shared_experts=1,
            top_k_experts=2,
            routing_function="softmax",
        ),
        adapter_hidden_dim=128,
    )
    return DeepSeekVL2Model(config)


def _smoke_test() -> None:
    """Runs a small forward pass to verify tensor shapes."""

    model = build_deepseek_vl2_tiny()
    images = [
        torch.randn(3, 40, 64),
        torch.randn(3, 64, 40),
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
    print("chosen_tile_grids:", output.chosen_tile_grids)
    print("num_tiles_per_image:", output.num_tiles_per_image)
    print("loss:", float(output.loss.detach()))


if __name__ == "__main__":
    _smoke_test()
