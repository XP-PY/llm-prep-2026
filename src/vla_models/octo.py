"""A compact educational Octo implementation in PyTorch.

This module follows the architecture summary in ``docs/VLAs/Octo.md``:

1. Language, goal image, RGB observation, and optional proprioceptive inputs
   are converted into a shared token space.
2. A transformer backbone processes task tokens, causal observation tokens, and
   learned readout tokens with a block-wise attention mask.
3. The final readout token conditions a small diffusion action head.
4. Training minimizes the DDPM noise-prediction objective on action chunks.
5. Inference samples continuous action chunks and executes the first few actions
   in a receding-horizon loop.

This implementation is intentionally educational rather than checkpoint-faithful:

* The language encoder is a trainable embedding table, not frozen T5-base.
* The image tokenizer is a small CNN + patch projection, not the official JAX
  model implementation.
* The diffusion action head is compact and self-contained for smoke tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Tensor:
    """Builds the squared-cosine beta schedule from improved DDPM."""

    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alpha_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


def _extract(values: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
    """Gathers timestep coefficients and reshapes them for broadcasting."""

    gathered = values.to(device=timesteps.device)[timesteps]
    return gathered.reshape(timesteps.shape[0], *([1] * (target_ndim - 1)))


def _best_group_count(channels: int, max_groups: int = 8) -> int:
    """Returns a GroupNorm group count that divides ``channels``."""

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding for diffusion timesteps."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embeds integer diffusion timesteps.

        Args:
            timesteps: Tensor with shape ``[B]``.

        Returns:
            Tensor with shape ``[B, width]``.
        """

        half_width = self.width // 2
        frequencies = torch.exp(
            torch.arange(half_width, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10_000.0) / max(half_width - 1, 1))
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.width:
            embedding = F.pad(embedding, (0, self.width - embedding.shape[-1]))
        return embedding


@dataclass(frozen=True)
class OctoObservationConfig:
    """Configuration for robot observations and action dimensions.

    Attributes:
        num_cameras: Number of RGB camera streams.
        image_height: Input image height used by this educational model.
        image_width: Input image width used by this educational model.
        in_channels: Number of image channels.
        observation_horizon: Number of observation frames in the context window.
        proprio_dim: Optional proprioceptive feature dimension per timestep.
        action_dim: Continuous robot action dimension.
        action_horizon: Number of future actions predicted per policy query.
        execution_horizon: Number of sampled actions executed before replanning.
        patch_size: Image patch size after the shallow CNN stem.
    """

    num_cameras: int = 2
    image_height: int = 128
    image_width: int = 128
    in_channels: int = 3
    observation_horizon: int = 2
    proprio_dim: int = 0
    action_dim: int = 7
    action_horizon: int = 8
    execution_horizon: int = 4
    patch_size: int = 16

    def __post_init__(self) -> None:
        if self.image_height % self.patch_size != 0 or self.image_width % self.patch_size != 0:
            raise ValueError("image_height and image_width must be divisible by patch_size.")
        if self.execution_horizon > self.action_horizon:
            raise ValueError("execution_horizon must be <= action_horizon.")

    @property
    def patches_per_image(self) -> int:
        """Returns the number of patch tokens per image."""

        return (self.image_height // self.patch_size) * (self.image_width // self.patch_size)


@dataclass(frozen=True)
class OctoTaskConfig:
    """Configuration for task-conditioning inputs."""

    vocab_size: int = 32_000
    max_language_tokens: int = 16
    max_goal_images: int = 1


@dataclass(frozen=True)
class OctoTransformerConfig:
    """Configuration for the Octo transformer backbone."""

    hidden_dim: int = 384
    layers: int = 12
    heads: int = 6
    mlp_dim: int = 1536
    dropout: float = 0.1


@dataclass(frozen=True)
class OctoDiffusionConfig:
    """Configuration for the diffusion action head."""

    num_train_timesteps: int = 20
    num_inference_steps: int = 10
    hidden_dim: int = 256
    time_embed_dim: int = 128
    layers: int = 3
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    clip_sample: bool = True


@dataclass(frozen=True)
class OctoConfig:
    """Top-level configuration for the educational Octo model."""

    observation_config: OctoObservationConfig = field(default_factory=OctoObservationConfig)
    task_config: OctoTaskConfig = field(default_factory=OctoTaskConfig)
    transformer_config: OctoTransformerConfig = field(default_factory=OctoTransformerConfig)
    diffusion_config: OctoDiffusionConfig = field(default_factory=OctoDiffusionConfig)


@dataclass
class OctoOutput:
    """Container returned by ``OctoModel.forward``.

    Attributes:
        token_embeddings: Transformer input tokens with shape ``[B, S, D]``.
        transformer_output: Transformer output tokens with shape ``[B, S, D]``.
        readout_embeddings: Readout embeddings with shape ``[B, T_o, D]``.
        latest_readout: Last readout embedding used by the action head.
        noisy_actions: Optional noisy action chunks used during training.
        predicted_noise: Optional predicted noise from the diffusion head.
        target_noise: Optional sampled Gaussian target noise.
        timesteps: Optional diffusion timesteps.
        loss: Optional DDPM MSE loss.
        sampled_actions: Optional sampled action chunk from inference.
    """

    token_embeddings: Tensor
    transformer_output: Tensor
    readout_embeddings: Tensor
    latest_readout: Tensor
    noisy_actions: Tensor | None = None
    predicted_noise: Tensor | None = None
    target_noise: Tensor | None = None
    timesteps: Tensor | None = None
    loss: Tensor | None = None
    sampled_actions: Tensor | None = None


class OctoImageTokenizer(nn.Module):
    """Shallow CNN plus patch tokenizer for goal and observation images."""

    def __init__(self, observation_config: OctoObservationConfig, hidden_dim: int) -> None:
        super().__init__()
        self.observation_config = observation_config
        self.hidden_dim = hidden_dim

        stem_width = max(hidden_dim // 4, 16)
        self.stem = nn.Sequential(
            nn.Conv2d(observation_config.in_channels, stem_width, 3, padding=1),
            nn.GroupNorm(_best_group_count(stem_width), stem_width),
            nn.GELU(),
            nn.Conv2d(stem_width, hidden_dim, 3, padding=1),
            nn.GroupNorm(_best_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.patch_projection = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=observation_config.patch_size,
            stride=observation_config.patch_size,
        )
        self.patch_position_embedding = nn.Parameter(
            torch.zeros(1, observation_config.patches_per_image, hidden_dim)
        )

    def forward(self, images: Tensor) -> Tensor:
        """Tokenizes a batch of images.

        Args:
            images: Tensor with shape ``[B, N, C, H, W]``.

        Returns:
            Patch tokens with shape ``[B, N, P, D]``.
        """

        if images.ndim != 5:
            raise ValueError("images must have shape [B, N, C, H, W].")

        batch_size, num_images, channels, height, width = images.shape
        expected = self.observation_config
        if channels != expected.in_channels:
            raise ValueError(f"Expected {expected.in_channels} image channels.")
        if height != expected.image_height or width != expected.image_width:
            raise ValueError(f"Expected image size {expected.image_height}x{expected.image_width}.")

        flat_images = images.reshape(batch_size * num_images, channels, height, width).float()
        features = self.stem(flat_images)
        patches = self.patch_projection(features)
        patch_tokens = patches.flatten(2).transpose(1, 2)
        patch_tokens = patch_tokens + self.patch_position_embedding
        return patch_tokens.reshape(batch_size, num_images, -1, self.hidden_dim)


class OctoDiffusionScheduler(nn.Module):
    """Stores DDPM coefficients and performs deterministic DDIM updates."""

    def __init__(self, config: OctoDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        if config.beta_schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
        elif config.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(config.num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule: {config.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )

    def add_noise(self, clean_actions: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Applies closed-form DDPM noising to action chunks."""

        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, clean_actions.ndim)
        sqrt_one_minus_alpha = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean_actions.ndim
        )
        return sqrt_alpha * clean_actions + sqrt_one_minus_alpha * noise

    def get_inference_timesteps(self, num_inference_steps: int | None, device: torch.device) -> Tensor:
        """Returns descending timesteps for shortened DDIM inference."""

        steps = num_inference_steps or self.config.num_inference_steps
        steps = max(1, min(steps, self.config.num_train_timesteps))
        return torch.linspace(
            self.config.num_train_timesteps - 1,
            0,
            steps,
            device=device,
            dtype=torch.long,
        )

    def ddim_step(
        self,
        predicted_noise: Tensor,
        timestep: Tensor,
        previous_timestep: int,
        sample: Tensor,
    ) -> Tensor:
        """Runs one deterministic DDIM denoising step."""

        alpha_bar_t = _extract(self.alphas_cumprod, timestep, sample.ndim)
        pred_original_sample = (sample - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(
            alpha_bar_t
        )
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        if previous_timestep < 0:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            previous = torch.full_like(timestep, previous_timestep)
            alpha_bar_prev = _extract(self.alphas_cumprod, previous, sample.ndim)

        return (
            torch.sqrt(alpha_bar_prev) * pred_original_sample
            + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
        )


class ResidualMLPBlock(nn.Module):
    """Small residual MLP block used inside the diffusion action head."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies a residual feed-forward update."""

        return x + self.net(x)


class OctoDiffusionActionHead(nn.Module):
    """Conditional diffusion head that predicts noise in action chunks."""

    def __init__(
        self,
        observation_config: OctoObservationConfig,
        transformer_config: OctoTransformerConfig,
        diffusion_config: OctoDiffusionConfig,
    ) -> None:
        super().__init__()
        self.observation_config = observation_config
        self.diffusion_config = diffusion_config

        hidden_dim = diffusion_config.hidden_dim
        self.time_embedding = SinusoidalTimeEmbedding(diffusion_config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_config.time_embed_dim, diffusion_config.time_embed_dim),
            nn.GELU(),
            nn.Linear(diffusion_config.time_embed_dim, diffusion_config.time_embed_dim),
        )
        self.action_projection = nn.Linear(observation_config.action_dim, hidden_dim)
        self.condition_projection = nn.Linear(
            transformer_config.hidden_dim + diffusion_config.time_embed_dim,
            hidden_dim,
        )
        self.action_position_embedding = nn.Parameter(
            torch.zeros(1, observation_config.action_horizon, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim) for _ in range(diffusion_config.layers)]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, observation_config.action_dim),
        )

    def forward(self, noisy_actions: Tensor, readout: Tensor, timesteps: Tensor) -> Tensor:
        """Predicts the Gaussian noise inside a noisy action chunk.

        Args:
            noisy_actions: Tensor with shape ``[B, H_a, A]``.
            readout: Transformer readout embedding with shape ``[B, D]``.
            timesteps: Diffusion timesteps with shape ``[B]``.

        Returns:
            Predicted noise with shape ``[B, H_a, A]``.
        """

        expected = self.observation_config
        if noisy_actions.shape[1:] != (expected.action_horizon, expected.action_dim):
            raise ValueError(
                "noisy_actions must have shape "
                f"[B, {expected.action_horizon}, {expected.action_dim}]."
            )

        time_features = self.time_mlp(self.time_embedding(timesteps))
        condition = self.condition_projection(torch.cat([readout, time_features], dim=-1))
        x = self.action_projection(noisy_actions)
        x = x + self.action_position_embedding + condition.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x)


class OctoModel(nn.Module):
    """Educational Octo model with flexible tokenization and diffusion actions."""

    TASK_ROLE = 0
    OBSERVATION_ROLE = 1
    READOUT_ROLE = 2

    def __init__(self, config: OctoConfig) -> None:
        super().__init__()
        self.config = config

        hidden_dim = config.transformer_config.hidden_dim
        self.language_embedding = nn.Embedding(config.task_config.vocab_size, hidden_dim)
        self.language_position_embedding = nn.Parameter(
            torch.zeros(1, config.task_config.max_language_tokens, hidden_dim)
        )
        self.null_task_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.image_tokenizer = OctoImageTokenizer(config.observation_config, hidden_dim)
        self.goal_index_embedding = nn.Embedding(config.task_config.max_goal_images, hidden_dim)
        self.camera_embedding = nn.Embedding(config.observation_config.num_cameras, hidden_dim)
        self.observation_time_embedding = nn.Embedding(
            config.observation_config.observation_horizon,
            hidden_dim,
        )
        self.modality_embedding = nn.Embedding(4, hidden_dim)
        self.readout_tokens = nn.Parameter(
            torch.zeros(1, config.observation_config.observation_horizon, hidden_dim)
        )

        if config.observation_config.proprio_dim > 0:
            self.proprio_projection = nn.Sequential(
                nn.LayerNorm(config.observation_config.proprio_dim),
                nn.Linear(config.observation_config.proprio_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.proprio_projection = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.transformer_config.heads,
            dim_feedforward=config.transformer_config.mlp_dim,
            dropout=config.transformer_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_config.layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.action_head = OctoDiffusionActionHead(
            config.observation_config,
            config.transformer_config,
            config.diffusion_config,
        )
        self.scheduler = OctoDiffusionScheduler(config.diffusion_config)

    def _tokenize_tasks(
        self,
        batch_size: int,
        device: torch.device,
        language_tokens: Tensor | None,
        language_mask: Tensor | None,
        goal_images: Tensor | None,
    ) -> tuple[Tensor, Tensor, list[int], list[int]]:
        """Builds task tokens from language tokens and optional goal images."""

        token_chunks: list[Tensor] = []
        valid_chunks: list[Tensor] = []
        roles: list[int] = []
        times: list[int] = []

        if language_tokens is not None:
            language_tokens = language_tokens[:, : self.config.task_config.max_language_tokens]
            lang_len = language_tokens.shape[1]
            language_features = self.language_embedding(language_tokens)
            language_features = language_features + self.language_position_embedding[:, :lang_len]
            language_features = language_features + self.modality_embedding.weight[0].view(1, 1, -1)
            token_chunks.append(language_features)
            if language_mask is None:
                valid_chunks.append(torch.ones(batch_size, lang_len, device=device, dtype=torch.bool))
            else:
                valid_chunks.append(language_mask[:, :lang_len].to(device=device, dtype=torch.bool))
            roles.extend([self.TASK_ROLE] * lang_len)
            times.extend([-1] * lang_len)

        if goal_images is not None:
            if goal_images.ndim != 5:
                raise ValueError("goal_images must have shape [B, G, C, H, W].")
            if goal_images.shape[1] > self.config.task_config.max_goal_images:
                raise ValueError("goal_images exceeds max_goal_images.")
            goal_tokens = self.image_tokenizer(goal_images)
            num_goals, patches = goal_tokens.shape[1], goal_tokens.shape[2]
            goal_ids = torch.arange(num_goals, device=device).view(1, num_goals, 1)
            goal_tokens = goal_tokens + self.goal_index_embedding(goal_ids)
            goal_tokens = goal_tokens + self.modality_embedding.weight[1].view(1, 1, 1, -1)
            goal_tokens = goal_tokens.reshape(batch_size, num_goals * patches, -1)
            token_chunks.append(goal_tokens)
            valid_chunks.append(
                torch.ones(batch_size, num_goals * patches, device=device, dtype=torch.bool)
            )
            roles.extend([self.TASK_ROLE] * (num_goals * patches))
            times.extend([-1] * (num_goals * patches))

        if not token_chunks:
            null_task = self.null_task_token.expand(batch_size, -1, -1)
            token_chunks.append(null_task)
            valid_chunks.append(torch.ones(batch_size, 1, device=device, dtype=torch.bool))
            roles.append(self.TASK_ROLE)
            times.append(-1)

        return (
            torch.cat(token_chunks, dim=1),
            torch.cat(valid_chunks, dim=1),
            roles,
            times,
        )

    def _encode_context(
        self,
        images: Tensor,
        language_tokens: Tensor | None,
        language_mask: Tensor | None,
        goal_images: Tensor | None,
        image_mask: Tensor | None,
        proprio: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Tokenizes inputs, applies block-wise attention, and returns readouts."""

        if images.ndim != 6:
            raise ValueError("images must have shape [B, T_o, Cams, C, H, W].")

        batch_size, obs_horizon, num_cameras, channels, height, width = images.shape
        obs_config = self.config.observation_config
        if obs_horizon != obs_config.observation_horizon:
            raise ValueError(f"Expected observation_horizon={obs_config.observation_horizon}.")
        if num_cameras != obs_config.num_cameras:
            raise ValueError(f"Expected num_cameras={obs_config.num_cameras}.")
        if channels != obs_config.in_channels:
            raise ValueError(f"Expected in_channels={obs_config.in_channels}.")
        if height != obs_config.image_height or width != obs_config.image_width:
            raise ValueError(f"Expected image size {obs_config.image_height}x{obs_config.image_width}.")

        device = images.device
        tokens, valid, roles, times = self._tokenize_tasks(
            batch_size=batch_size,
            device=device,
            language_tokens=language_tokens,
            language_mask=language_mask,
            goal_images=goal_images,
        )
        token_chunks = [tokens]
        valid_chunks = [valid]
        readout_indices: list[int] = []

        flat_images = images.reshape(
            batch_size,
            obs_horizon * num_cameras,
            channels,
            height,
            width,
        )
        image_tokens = self.image_tokenizer(flat_images)
        patches = image_tokens.shape[2]
        image_tokens = image_tokens.reshape(batch_size, obs_horizon, num_cameras, patches, -1)

        if image_mask is None:
            image_mask = torch.ones(batch_size, obs_horizon, num_cameras, device=device, dtype=torch.bool)
        else:
            image_mask = image_mask.to(device=device, dtype=torch.bool)

        for time_idx in range(obs_horizon):
            camera_ids = torch.arange(num_cameras, device=device).view(1, num_cameras, 1)
            time_id = torch.full((1, 1, 1), time_idx, device=device, dtype=torch.long)
            obs_tokens = image_tokens[:, time_idx]
            obs_tokens = obs_tokens + self.camera_embedding(camera_ids)
            obs_tokens = obs_tokens + self.observation_time_embedding(time_id)
            obs_tokens = obs_tokens + self.modality_embedding.weight[2].view(1, 1, 1, -1)
            obs_tokens = obs_tokens.reshape(batch_size, num_cameras * patches, -1)

            obs_valid = image_mask[:, time_idx].unsqueeze(-1).expand(batch_size, num_cameras, patches)
            obs_valid = obs_valid.reshape(batch_size, num_cameras * patches)
            token_chunks.append(obs_tokens)
            valid_chunks.append(obs_valid)
            roles.extend([self.OBSERVATION_ROLE] * (num_cameras * patches))
            times.extend([time_idx] * (num_cameras * patches))

            if self.proprio_projection is not None and proprio is not None:
                proprio_token = self.proprio_projection(proprio[:, time_idx]).unsqueeze(1)
                proprio_token = proprio_token + self.observation_time_embedding.weight[time_idx].view(
                    1,
                    1,
                    -1,
                )
                proprio_token = proprio_token + self.modality_embedding.weight[3].view(1, 1, -1)
                token_chunks.append(proprio_token)
                valid_chunks.append(torch.ones(batch_size, 1, device=device, dtype=torch.bool))
                roles.append(self.OBSERVATION_ROLE)
                times.append(time_idx)

            readout_index = sum(chunk.shape[1] for chunk in token_chunks)
            readout = self.readout_tokens[:, time_idx : time_idx + 1].expand(batch_size, -1, -1)
            readout = readout + self.observation_time_embedding.weight[time_idx].view(1, 1, -1)
            token_chunks.append(readout)
            valid_chunks.append(torch.ones(batch_size, 1, device=device, dtype=torch.bool))
            roles.append(self.READOUT_ROLE)
            times.append(time_idx)
            readout_indices.append(readout_index)

        token_embeddings = torch.cat(token_chunks, dim=1)
        valid_tokens = torch.cat(valid_chunks, dim=1)
        role_tensor = torch.tensor(roles, device=device, dtype=torch.long)
        time_tensor = torch.tensor(times, device=device, dtype=torch.long)
        attention_mask = self._build_block_attention_mask(role_tensor, time_tensor)

        transformer_output = self.transformer(
            token_embeddings,
            mask=attention_mask,
            src_key_padding_mask=~valid_tokens,
        )
        readout_embeddings = transformer_output[:, readout_indices]
        return token_embeddings, transformer_output, readout_embeddings, valid_tokens

    def _build_block_attention_mask(self, roles: Tensor, times: Tensor) -> Tensor:
        """Builds Octo-style block-wise causal attention.

        ``True`` entries are masked out, matching PyTorch's transformer API.
        """

        seq_len = roles.shape[0]
        mask = torch.ones(seq_len, seq_len, device=roles.device, dtype=torch.bool)
        for query_idx in range(seq_len):
            query_role = int(roles[query_idx].item())
            query_time = int(times[query_idx].item())
            for key_idx in range(seq_len):
                key_role = int(roles[key_idx].item())
                key_time = int(times[key_idx].item())

                if query_role == self.TASK_ROLE:
                    allowed = key_role == self.TASK_ROLE
                elif query_role == self.OBSERVATION_ROLE:
                    allowed = key_role == self.TASK_ROLE or (
                        key_role == self.OBSERVATION_ROLE and key_time <= query_time
                    )
                else:
                    allowed = (
                        key_role == self.TASK_ROLE
                        or (key_role == self.OBSERVATION_ROLE and key_time <= query_time)
                        or query_idx == key_idx
                    )

                if key_role == self.READOUT_ROLE and query_idx != key_idx:
                    allowed = False
                mask[query_idx, key_idx] = not allowed
        return mask

    def forward(
        self,
        images: Tensor,
        language_tokens: Tensor | None = None,
        language_mask: Tensor | None = None,
        goal_images: Tensor | None = None,
        image_mask: Tensor | None = None,
        proprio: Tensor | None = None,
        action_sequence: Tensor | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> OctoOutput:
        """Runs Octo context encoding and optional diffusion training objective."""

        token_embeddings, transformer_output, readout_embeddings, _ = self._encode_context(
            images=images,
            language_tokens=language_tokens,
            language_mask=language_mask,
            goal_images=goal_images,
            image_mask=image_mask,
            proprio=proprio,
        )
        latest_readout = readout_embeddings[:, -1]

        if action_sequence is None:
            return OctoOutput(
                token_embeddings=token_embeddings,
                transformer_output=transformer_output,
                readout_embeddings=readout_embeddings,
                latest_readout=latest_readout,
            )

        batch_size = action_sequence.shape[0]
        expected = self.config.observation_config
        if action_sequence.shape != (batch_size, expected.action_horizon, expected.action_dim):
            raise ValueError(
                "action_sequence must have shape "
                f"[B, {expected.action_horizon}, {expected.action_dim}]."
            )
        if noise is None:
            noise = torch.randn_like(action_sequence)
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.config.diffusion_config.num_train_timesteps,
                size=(batch_size,),
                device=action_sequence.device,
            )

        noisy_actions = self.scheduler.add_noise(action_sequence, noise, timesteps)
        predicted_noise = self.action_head(noisy_actions, latest_readout, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        return OctoOutput(
            token_embeddings=token_embeddings,
            transformer_output=transformer_output,
            readout_embeddings=readout_embeddings,
            latest_readout=latest_readout,
            noisy_actions=noisy_actions,
            predicted_noise=predicted_noise,
            target_noise=noise,
            timesteps=timesteps,
            loss=loss,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        images: Tensor,
        language_tokens: Tensor | None = None,
        language_mask: Tensor | None = None,
        goal_images: Tensor | None = None,
        image_mask: Tensor | None = None,
        proprio: Tensor | None = None,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Samples a full future action chunk with DDIM-style denoising."""

        _, _, readout_embeddings, _ = self._encode_context(
            images=images,
            language_tokens=language_tokens,
            language_mask=language_mask,
            goal_images=goal_images,
            image_mask=image_mask,
            proprio=proprio,
        )
        readout = readout_embeddings[:, -1]
        batch_size = images.shape[0]
        obs_config = self.config.observation_config
        sample = torch.randn(
            batch_size,
            obs_config.action_horizon,
            obs_config.action_dim,
            device=images.device,
            generator=generator,
        )

        timesteps = self.scheduler.get_inference_timesteps(num_inference_steps, images.device)
        for index, timestep in enumerate(timesteps):
            timestep_batch = torch.full(
                (batch_size,),
                int(timestep.item()),
                device=images.device,
                dtype=torch.long,
            )
            predicted_noise = self.action_head(sample, readout, timestep_batch)
            previous_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            sample = self.scheduler.ddim_step(
                predicted_noise=predicted_noise,
                timestep=timestep_batch,
                previous_timestep=previous_timestep,
                sample=sample,
            )
        return sample

    @torch.no_grad()
    def predict_action(
        self,
        images: Tensor,
        language_tokens: Tensor | None = None,
        language_mask: Tensor | None = None,
        goal_images: Tensor | None = None,
        image_mask: Tensor | None = None,
        proprio: Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        """Returns the receding-horizon action prefix to execute now."""

        actions = self.sample_actions(
            images=images,
            language_tokens=language_tokens,
            language_mask=language_mask,
            goal_images=goal_images,
            image_mask=image_mask,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
        )
        return actions[:, : self.config.observation_config.execution_horizon]


def build_octo_tiny() -> OctoModel:
    """Builds a small Octo model suitable for CPU smoke tests."""

    config = OctoConfig(
        observation_config=OctoObservationConfig(
            num_cameras=2,
            image_height=64,
            image_width=64,
            in_channels=3,
            observation_horizon=2,
            proprio_dim=4,
            action_dim=5,
            action_horizon=6,
            execution_horizon=2,
            patch_size=16,
        ),
        task_config=OctoTaskConfig(
            vocab_size=128,
            max_language_tokens=8,
            max_goal_images=1,
        ),
        transformer_config=OctoTransformerConfig(
            hidden_dim=64,
            layers=2,
            heads=4,
            mlp_dim=128,
            dropout=0.0,
        ),
        diffusion_config=OctoDiffusionConfig(
            num_train_timesteps=12,
            num_inference_steps=4,
            hidden_dim=64,
            time_embed_dim=64,
            layers=2,
            beta_schedule="cosine",
        ),
    )
    return OctoModel(config)


def _smoke_test() -> None:
    """Runs one training-style pass and one inference-style pass."""

    torch.manual_seed(17)
    model = build_octo_tiny()
    obs_config = model.config.observation_config
    batch_size = 2

    images = torch.randn(
        batch_size,
        obs_config.observation_horizon,
        obs_config.num_cameras,
        obs_config.in_channels,
        obs_config.image_height,
        obs_config.image_width,
    )
    language_tokens = torch.randint(0, model.config.task_config.vocab_size, (batch_size, 8))
    language_mask = torch.ones(batch_size, 8, dtype=torch.bool)
    goal_images = torch.randn(
        batch_size,
        1,
        obs_config.in_channels,
        obs_config.image_height,
        obs_config.image_width,
    )
    proprio = torch.randn(batch_size, obs_config.observation_horizon, obs_config.proprio_dim)
    action_sequence = torch.randn(batch_size, obs_config.action_horizon, obs_config.action_dim).clamp(
        -1.0,
        1.0,
    )

    output = model(
        images=images,
        language_tokens=language_tokens,
        language_mask=language_mask,
        goal_images=goal_images,
        proprio=proprio,
        action_sequence=action_sequence,
    )
    model.eval()
    sampled_actions = model.sample_actions(
        images=images,
        language_tokens=language_tokens,
        language_mask=language_mask,
        goal_images=goal_images,
        proprio=proprio,
        num_inference_steps=4,
    )
    action_prefix = model.predict_action(
        images=images,
        language_tokens=language_tokens,
        language_mask=language_mask,
        goal_images=goal_images,
        proprio=proprio,
        num_inference_steps=4,
    )

    print(f"training_loss={output.loss.item():.4f}")
    print(f"token_embeddings_shape={tuple(output.token_embeddings.shape)}")
    print(f"readout_embeddings_shape={tuple(output.readout_embeddings.shape)}")
    print(f"predicted_noise_shape={tuple(output.predicted_noise.shape)}")
    print(f"sampled_actions_shape={tuple(sampled_actions.shape)}")
    print(f"receding_horizon_action_prefix_shape={tuple(action_prefix.shape)}")


if __name__ == "__main__":
    _smoke_test()
