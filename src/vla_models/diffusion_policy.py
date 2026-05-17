"""A compact educational Diffusion Policy implementation in PyTorch.

This module follows the architecture summary in ``docs/VLAs/Diffusion_Policy.md``:

1. Recent RGB observations and optional proprioception are encoded once.
2. A temporal action denoiser receives noisy future action sequences.
3. Observation features and diffusion-step embeddings modulate the denoiser.
4. Training minimizes the DDPM noise-prediction objective.
5. Inference starts from Gaussian action noise and uses DDIM-style denoising.
6. Receding-horizon deployment executes only the first few generated actions.

The implementation is intentionally educational rather than checkpoint-faithful:

* The visual encoder is a small shared CNN, not the paper's ResNet18.
* The denoiser is a readable FiLM-conditioned temporal ConvNet.
* Sampling uses deterministic DDIM updates for fast smoke tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _best_group_count(channels: int, max_groups: int = 8) -> int:
    """Returns a GroupNorm group count that divides ``channels``."""

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Tensor:
    """Builds the squared-cosine beta schedule used by improved DDPM."""

    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alpha_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1.0 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


def _extract(values: Tensor, timesteps: Tensor, target_ndim: int) -> Tensor:
    """Gathers timestep values and reshapes them for broadcasting."""

    gathered = values.to(device=timesteps.device)[timesteps]
    return gathered.reshape(timesteps.shape[0], *([1] * (target_ndim - 1)))


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
class DiffusionPolicyObservationConfig:
    """Configuration for observation and action dimensions.

    Attributes:
        num_cameras: Number of RGB camera views.
        image_height: Input image height used by the smoke-test model.
        image_width: Input image width used by the smoke-test model.
        in_channels: Number of image channels.
        proprio_dim: Dimension of proprioceptive observations per timestep.
        action_dim: Dimension of continuous robot action per timestep.
        observation_horizon: Number of recent observation steps ``T_o``.
        prediction_horizon: Number of future actions generated ``T_p``.
        execution_horizon: Number of generated actions executed before replanning ``T_a``.
    """

    num_cameras: int = 2
    image_height: int = 96
    image_width: int = 96
    in_channels: int = 3
    proprio_dim: int = 14
    action_dim: int = 14
    observation_horizon: int = 2
    prediction_horizon: int = 16
    execution_horizon: int = 8

    def __post_init__(self) -> None:
        if self.execution_horizon > self.prediction_horizon:
            raise ValueError("execution_horizon must be <= prediction_horizon.")


@dataclass(frozen=True)
class DiffusionPolicyVisionConfig:
    """Configuration for the compact visual / proprioceptive encoder."""

    stem_width: int = 32
    visual_width: int = 128
    context_dim: int = 256
    dropout: float = 0.0


@dataclass(frozen=True)
class DiffusionPolicyDenoiserConfig:
    """Configuration for the FiLM-conditioned temporal denoiser."""

    model_width: int = 256
    time_embed_dim: int = 256
    hidden_layers: int = 6
    kernel_size: int = 5
    dropout: float = 0.0


@dataclass(frozen=True)
class DiffusionPolicySchedulerConfig:
    """Configuration for DDPM training noise and DDIM inference."""

    num_train_timesteps: int = 100
    num_inference_steps: int = 16
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: Literal["linear", "cosine"] = "cosine"
    clip_sample: bool = True


@dataclass(frozen=True)
class DiffusionPolicyConfig:
    """Top-level configuration for the educational Diffusion Policy model."""

    observation_config: DiffusionPolicyObservationConfig = field(
        default_factory=DiffusionPolicyObservationConfig
    )
    vision_config: DiffusionPolicyVisionConfig = field(default_factory=DiffusionPolicyVisionConfig)
    denoiser_config: DiffusionPolicyDenoiserConfig = field(
        default_factory=DiffusionPolicyDenoiserConfig
    )
    scheduler_config: DiffusionPolicySchedulerConfig = field(
        default_factory=DiffusionPolicySchedulerConfig
    )


@dataclass
class DiffusionPolicyOutput:
    """Container returned by ``DiffusionPolicyModel.forward``.

    Attributes:
        observation_embedding: Encoded observation context with shape ``[B, D]``.
        noisy_actions: Noisy training actions with shape ``[B, T_p, A]``.
        predicted_noise: Denoiser output with shape ``[B, T_p, A]``.
        target_noise: Gaussian training target with shape ``[B, T_p, A]``.
        timesteps: Sampled diffusion timesteps with shape ``[B]``.
        loss: Optional MSE noise-prediction loss.
        sampled_actions: Optional denoised action sequence from inference.
    """

    observation_embedding: Tensor
    noisy_actions: Tensor | None = None
    predicted_noise: Tensor | None = None
    target_noise: Tensor | None = None
    timesteps: Tensor | None = None
    loss: Tensor | None = None
    sampled_actions: Tensor | None = None


class MinMaxActionNormalizer:
    """Normalizes continuous actions to the ``[-1, 1]`` range used by diffusion."""

    def __init__(self, action_min: Tensor, action_max: Tensor, eps: float = 1e-6) -> None:
        self.action_min = action_min
        self.action_max = action_max
        self.eps = eps

    def normalize(self, actions: Tensor) -> Tensor:
        """Scales raw actions to ``[-1, 1]`` dimension-wise."""

        action_min = self.action_min.to(device=actions.device, dtype=actions.dtype)
        action_max = self.action_max.to(device=actions.device, dtype=actions.dtype)
        return 2.0 * (actions - action_min) / (action_max - action_min + self.eps) - 1.0

    def denormalize(self, actions: Tensor) -> Tensor:
        """Maps normalized actions back to the original control units."""

        action_min = self.action_min.to(device=actions.device, dtype=actions.dtype)
        action_max = self.action_max.to(device=actions.device, dtype=actions.dtype)
        return 0.5 * (actions + 1.0) * (action_max - action_min) + action_min


class DiffusionNoiseScheduler(nn.Module):
    """Stores DDPM coefficients and performs fast deterministic DDIM updates."""

    def __init__(self, config: DiffusionPolicySchedulerConfig) -> None:
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

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )

    def add_noise(self, clean_actions: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Applies the closed-form DDPM forward process to clean action chunks."""

        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, clean_actions.ndim)
        sqrt_one_minus_alpha = _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean_actions.ndim
        )
        return sqrt_alpha * clean_actions + sqrt_one_minus_alpha * noise

    def get_inference_timesteps(self, num_inference_steps: int | None, device: torch.device) -> Tensor:
        """Returns descending timesteps used by DDIM inference."""

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
        """Runs one deterministic DDIM update.

        Args:
            predicted_noise: Noise predicted by the denoiser.
            timestep: Current diffusion timestep, shape ``[B]``.
            previous_timestep: Next timestep in the shortened inference schedule.
            sample: Current noisy action sequence.

        Returns:
            Less noisy action sequence with the same shape as ``sample``.
        """

        alpha_bar_t = _extract(self.alphas_cumprod, timestep, sample.ndim)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        pred_original_sample = (sample - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

        if previous_timestep < 0:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            previous = torch.full_like(timestep, previous_timestep)
            alpha_bar_prev = _extract(self.alphas_cumprod, previous, sample.ndim)

        prev_sample = (
            torch.sqrt(alpha_bar_prev) * pred_original_sample
            + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
        )
        return prev_sample


class DiffusionPolicyObservationEncoder(nn.Module):
    """Encodes recent multi-camera images and optional proprioception once per query."""

    def __init__(
        self,
        observation_config: DiffusionPolicyObservationConfig,
        vision_config: DiffusionPolicyVisionConfig,
    ) -> None:
        super().__init__()
        self.observation_config = observation_config
        self.vision_config = vision_config

        stem = vision_config.stem_width
        visual_width = vision_config.visual_width
        context_dim = vision_config.context_dim

        self.image_encoder = nn.Sequential(
            nn.Conv2d(observation_config.in_channels, stem, 5, stride=2, padding=2),
            nn.GroupNorm(_best_group_count(stem), stem),
            nn.GELU(),
            nn.Conv2d(stem, stem, 3, stride=2, padding=1),
            nn.GroupNorm(_best_group_count(stem), stem),
            nn.GELU(),
            nn.Conv2d(stem, visual_width, 3, stride=2, padding=1),
            nn.GroupNorm(_best_group_count(visual_width), visual_width),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.camera_embedding = nn.Embedding(observation_config.num_cameras, visual_width)
        self.observation_time_embedding = nn.Embedding(
            observation_config.observation_horizon, visual_width
        )
        self.image_projection = nn.Sequential(
            nn.LayerNorm(visual_width),
            nn.Linear(visual_width, context_dim),
            nn.GELU(),
            nn.Dropout(vision_config.dropout),
        )

        if observation_config.proprio_dim > 0:
            self.proprio_projection = nn.Sequential(
                nn.LayerNorm(observation_config.proprio_dim),
                nn.Linear(observation_config.proprio_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim),
            )
        else:
            self.proprio_projection = None

        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

    def forward(self, images: Tensor, proprio: Tensor | None = None) -> Tensor:
        """Encodes observations.

        Args:
            images: RGB tensor with shape ``[B, T_o, Cams, C, H, W]``.
            proprio: Optional proprioception with shape ``[B, T_o, P]``.

        Returns:
            Observation context vector with shape ``[B, context_dim]``.
        """

        if images.ndim != 6:
            raise ValueError("images must have shape [B, T_o, Cams, C, H, W].")

        batch_size, obs_horizon, num_cameras, channels, height, width = images.shape
        expected = self.observation_config
        if obs_horizon != expected.observation_horizon:
            raise ValueError(f"Expected observation_horizon={expected.observation_horizon}.")
        if num_cameras != expected.num_cameras:
            raise ValueError(f"Expected num_cameras={expected.num_cameras}.")
        if channels != expected.in_channels:
            raise ValueError(f"Expected in_channels={expected.in_channels}.")

        # Images are encoded independently per time and camera, then pooled into a
        # single conditioning vector. This mirrors the paper's "encode once, denoise
        # actions many times" design.
        flat_images = images.reshape(batch_size * obs_horizon * num_cameras, channels, height, width)
        flat_features = self.image_encoder(flat_images.float())
        image_features = flat_features.reshape(batch_size, obs_horizon, num_cameras, -1)

        camera_ids = torch.arange(num_cameras, device=images.device).reshape(1, 1, num_cameras)
        time_ids = torch.arange(obs_horizon, device=images.device).reshape(1, obs_horizon, 1)
        image_features = (
            image_features
            + self.camera_embedding(camera_ids)
            + self.observation_time_embedding(time_ids)
        )
        image_tokens = self.image_projection(image_features)
        context = image_tokens.mean(dim=(1, 2))

        if self.proprio_projection is not None:
            if proprio is None:
                raise ValueError("proprio must be provided when proprio_dim > 0.")
            if proprio.shape != (
                batch_size,
                obs_horizon,
                self.observation_config.proprio_dim,
            ):
                raise ValueError(
                    "proprio must have shape "
                    f"[B, {obs_horizon}, {self.observation_config.proprio_dim}]."
                )
            proprio_tokens = self.proprio_projection(proprio.float())
            context = context + proprio_tokens.mean(dim=1)

        return self.context_projection(context)


class ConditionalResidualBlock1D(nn.Module):
    """FiLM-conditioned temporal residual block for action sequences."""

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(_best_group_count(channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(_best_group_count(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.film = nn.Sequential(
            nn.GELU(),
            nn.Linear(condition_dim, channels * 2),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Applies a conditioned residual update.

        Args:
            x: Temporal features with shape ``[B, C, T_p]``.
            condition: Observation-plus-timestep embedding with shape ``[B, D]``.

        Returns:
            Updated temporal features with shape ``[B, C, T_p]``.
        """

        scale, shift = self.film(condition).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.gelu(h)))
        return x + h


class TemporalActionDenoiser(nn.Module):
    """Predicts Gaussian noise in a future action sequence."""

    def __init__(
        self,
        observation_config: DiffusionPolicyObservationConfig,
        vision_config: DiffusionPolicyVisionConfig,
        denoiser_config: DiffusionPolicyDenoiserConfig,
    ) -> None:
        super().__init__()
        self.observation_config = observation_config
        self.denoiser_config = denoiser_config

        model_width = denoiser_config.model_width
        condition_dim = vision_config.context_dim + denoiser_config.time_embed_dim

        self.time_embedding = SinusoidalTimeEmbedding(denoiser_config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(denoiser_config.time_embed_dim, denoiser_config.time_embed_dim),
            nn.GELU(),
            nn.Linear(denoiser_config.time_embed_dim, denoiser_config.time_embed_dim),
        )
        self.input_projection = nn.Conv1d(observation_config.action_dim, model_width, 1)
        self.action_position_embedding = nn.Parameter(
            torch.zeros(1, model_width, observation_config.prediction_horizon)
        )

        dilations = [2 ** (idx % 4) for idx in range(denoiser_config.hidden_layers)]
        self.blocks = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    channels=model_width,
                    condition_dim=condition_dim,
                    kernel_size=denoiser_config.kernel_size,
                    dilation=dilation,
                    dropout=denoiser_config.dropout,
                )
                for dilation in dilations
            ]
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(_best_group_count(model_width), model_width),
            nn.GELU(),
            nn.Conv1d(model_width, observation_config.action_dim, 1),
        )

    def forward(self, noisy_actions: Tensor, timesteps: Tensor, observation_embedding: Tensor) -> Tensor:
        """Predicts the noise component of ``noisy_actions``.

        Args:
            noisy_actions: Tensor with shape ``[B, T_p, A]``.
            timesteps: Diffusion timesteps with shape ``[B]``.
            observation_embedding: Encoded observation context with shape ``[B, D]``.

        Returns:
            Predicted noise with shape ``[B, T_p, A]``.
        """

        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape [B, T_p, A].")

        if noisy_actions.shape[1] != self.observation_config.prediction_horizon:
            raise ValueError(
                f"Expected prediction_horizon={self.observation_config.prediction_horizon}."
            )
        if noisy_actions.shape[2] != self.observation_config.action_dim:
            raise ValueError(f"Expected action_dim={self.observation_config.action_dim}.")

        time_embedding = self.time_mlp(self.time_embedding(timesteps))
        condition = torch.cat([observation_embedding, time_embedding], dim=-1)

        x = noisy_actions.transpose(1, 2)
        x = self.input_projection(x) + self.action_position_embedding
        for block in self.blocks:
            x = block(x, condition)

        predicted_noise = self.output_projection(x)
        return predicted_noise.transpose(1, 2)


class DiffusionPolicyModel(nn.Module):
    """Educational Diffusion Policy for visuomotor action-sequence generation."""

    def __init__(self, config: DiffusionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.observation_encoder = DiffusionPolicyObservationEncoder(
            config.observation_config,
            config.vision_config,
        )
        self.denoiser = TemporalActionDenoiser(
            config.observation_config,
            config.vision_config,
            config.denoiser_config,
        )
        self.scheduler = DiffusionNoiseScheduler(config.scheduler_config)

    def forward(
        self,
        images: Tensor,
        proprio: Tensor | None = None,
        action_sequence: Tensor | None = None,
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> DiffusionPolicyOutput:
        """Runs the training objective when ``action_sequence`` is provided.

        Args:
            images: Observation images with shape ``[B, T_o, Cams, C, H, W]``.
            proprio: Optional proprioception with shape ``[B, T_o, P]``.
            action_sequence: Optional clean future actions with shape ``[B, T_p, A]``.
            noise: Optional Gaussian noise target for deterministic tests.
            timesteps: Optional diffusion timesteps for deterministic tests.

        Returns:
            ``DiffusionPolicyOutput`` containing embeddings and optional loss tensors.
        """

        observation_embedding = self.observation_encoder(images, proprio)
        if action_sequence is None:
            return DiffusionPolicyOutput(observation_embedding=observation_embedding)

        batch_size = action_sequence.shape[0]
        expected_horizon = self.config.observation_config.prediction_horizon
        expected_action_dim = self.config.observation_config.action_dim
        if action_sequence.shape != (batch_size, expected_horizon, expected_action_dim):
            raise ValueError(
                "action_sequence must have shape "
                f"[B, {expected_horizon}, {expected_action_dim}]."
            )

        if noise is None:
            noise = torch.randn_like(action_sequence)
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.config.scheduler_config.num_train_timesteps,
                size=(batch_size,),
                device=action_sequence.device,
            )

        noisy_actions = self.scheduler.add_noise(action_sequence, noise, timesteps)
        predicted_noise = self.denoiser(noisy_actions, timesteps, observation_embedding)
        loss = F.mse_loss(predicted_noise, noise)

        return DiffusionPolicyOutput(
            observation_embedding=observation_embedding,
            noisy_actions=noisy_actions,
            predicted_noise=predicted_noise,
            target_noise=noise,
            timesteps=timesteps,
            loss=loss,
        )

    @torch.no_grad()
    def sample_action_sequence(
        self,
        images: Tensor,
        proprio: Tensor | None = None,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Samples a full future action sequence with DDIM-style denoising."""

        observation_embedding = self.observation_encoder(images, proprio)
        batch_size = images.shape[0]
        obs_config = self.config.observation_config
        sample = torch.randn(
            batch_size,
            obs_config.prediction_horizon,
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
            predicted_noise = self.denoiser(sample, timestep_batch, observation_embedding)
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
        proprio: Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> Tensor:
        """Returns the receding-horizon action chunk to execute now."""

        action_sequence = self.sample_action_sequence(
            images=images,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
        )
        execution_horizon = self.config.observation_config.execution_horizon
        return action_sequence[:, :execution_horizon]


def build_diffusion_policy_tiny() -> DiffusionPolicyModel:
    """Builds a small model suitable for CPU smoke tests."""

    config = DiffusionPolicyConfig(
        observation_config=DiffusionPolicyObservationConfig(
            num_cameras=2,
            image_height=64,
            image_width=64,
            in_channels=3,
            proprio_dim=8,
            action_dim=6,
            observation_horizon=2,
            prediction_horizon=8,
            execution_horizon=3,
        ),
        vision_config=DiffusionPolicyVisionConfig(
            stem_width=16,
            visual_width=32,
            context_dim=64,
            dropout=0.0,
        ),
        denoiser_config=DiffusionPolicyDenoiserConfig(
            model_width=64,
            time_embed_dim=64,
            hidden_layers=3,
            kernel_size=5,
            dropout=0.0,
        ),
        scheduler_config=DiffusionPolicySchedulerConfig(
            num_train_timesteps=16,
            num_inference_steps=4,
            beta_schedule="cosine",
        ),
    )
    return DiffusionPolicyModel(config)


def _smoke_test() -> None:
    """Runs one training-style pass and one inference-style pass."""

    torch.manual_seed(7)
    model = build_diffusion_policy_tiny()

    batch_size = 2
    obs_config = model.config.observation_config
    images = torch.randn(
        batch_size,
        obs_config.observation_horizon,
        obs_config.num_cameras,
        obs_config.in_channels,
        obs_config.image_height,
        obs_config.image_width,
    )
    proprio = torch.randn(batch_size, obs_config.observation_horizon, obs_config.proprio_dim)
    action_sequence = torch.randn(
        batch_size,
        obs_config.prediction_horizon,
        obs_config.action_dim,
    ).clamp(-1.0, 1.0)

    output = model(images=images, proprio=proprio, action_sequence=action_sequence)
    model.eval()
    sampled_sequence = model.sample_action_sequence(images=images, proprio=proprio, num_inference_steps=4)
    action_chunk = model.predict_action(images=images, proprio=proprio, num_inference_steps=4)

    print(f"training_loss={output.loss.item():.4f}")
    print(f"predicted_noise_shape={tuple(output.predicted_noise.shape)}")
    print(f"sampled_sequence_shape={tuple(sampled_sequence.shape)}")
    print(f"receding_horizon_action_chunk_shape={tuple(action_chunk.shape)}")


if __name__ == "__main__":
    _smoke_test()
