"""A compact educational ACT implementation in PyTorch.

This module follows the architecture summary in ``docs/VLAs/ACT.md``:

1. Four RGB camera streams are converted into visual tokens.
2. Joint positions and a latent style variable are appended to the visual
   context.
3. A Transformer encoder fuses camera, joint, and style information.
4. A Transformer decoder predicts a chunk of future continuous joint targets.
5. During training, a CVAE encoder infers the style variable from the target
   action chunk and current joints; during inference, the style variable is set
   to zero.
6. A temporal ensemble helper combines overlapping action chunks at deployment.

This implementation is intentionally educational rather than checkpoint-faithful:

* The image backbone is a small CNN, not the ResNet18 used in the paper.
* The Transformer stack is compact and configurable.
* The temporal ensemble is implemented as a readable Python buffer instead of a
  real-time robotics controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _build_1d_sincos_position_embedding(length: int, width: int) -> Tensor:
    """Builds fixed 1D sinusoidal position embeddings.

    Args:
        length: Number of sequence positions.
        width: Embedding width. Must be even.

    Returns:
        Tensor with shape ``[length, width]``.
    """

    if width % 2 != 0:
        raise ValueError("width must be even for sinusoidal embeddings.")

    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width)
    )
    embedding = torch.zeros(length, width, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(position * div_term)
    embedding[:, 1::2] = torch.cos(position * div_term)
    return embedding


def _build_2d_sincos_position_embedding(height: int, width: int, channels: int) -> Tensor:
    """Builds fixed 2D sinusoidal embeddings for a feature grid.

    Args:
        height: Feature-grid height.
        width: Feature-grid width.
        channels: Embedding width. Must be divisible by four.

    Returns:
        Tensor with shape ``[height * width, channels]``.
    """

    if channels % 4 != 0:
        raise ValueError("channels must be divisible by 4 for 2D sinusoidal embeddings.")

    y_embedding = _build_1d_sincos_position_embedding(height, channels // 2)
    x_embedding = _build_1d_sincos_position_embedding(width, channels // 2)

    y_grid = y_embedding[:, None, :].expand(height, width, channels // 2)
    x_grid = x_embedding[None, :, :].expand(height, width, channels // 2)
    position_embedding = torch.cat([y_grid, x_grid], dim=-1)
    return position_embedding.reshape(height * width, channels)


@dataclass(frozen=True)
class ACTObservationConfig:
    """Configuration for ACT observations and action dimensions.

    Attributes:
        num_cameras: Number of RGB camera views.
        image_height: Input image height.
        image_width: Input image width.
        in_channels: Number of image channels.
        joint_dim: Number of current robot joint-position inputs.
        action_dim: Number of target joint-position outputs.
        feature_grid_height: Height of the visual feature grid per camera.
        feature_grid_width: Width of the visual feature grid per camera.
    """

    num_cameras: int = 4
    image_height: int = 480
    image_width: int = 640
    in_channels: int = 3
    joint_dim: int = 14
    action_dim: int = 14
    feature_grid_height: int = 15
    feature_grid_width: int = 20

    @property
    def tokens_per_camera(self) -> int:
        """Returns the flattened visual token count per camera."""

        return self.feature_grid_height * self.feature_grid_width

    @property
    def total_visual_tokens(self) -> int:
        """Returns the visual token count across all cameras."""

        return self.num_cameras * self.tokens_per_camera


@dataclass(frozen=True)
class ACTVisionConfig:
    """Configuration for the compact CNN image tokenizer."""

    stem_width: int = 64
    visual_width: int = 512
    dropout: float = 0.0


@dataclass(frozen=True)
class ACTTransformerConfig:
    """Configuration for ACT's Transformer and latent style variable.

    Attributes:
        hidden_dim: Transformer hidden width.
        latent_dim: Dimension of the CVAE style variable.
        chunk_size: Number of future actions predicted per policy query.
        encoder_layers: Number of context Transformer encoder layers.
        decoder_layers: Number of action Transformer decoder layers.
        cvae_encoder_layers: Number of CVAE style encoder layers.
        heads: Number of attention heads.
        feedforward_dim: Feed-forward width in Transformer blocks.
        dropout: Dropout used inside Transformer layers.
    """

    hidden_dim: int = 512
    latent_dim: int = 32
    chunk_size: int = 100
    encoder_layers: int = 4
    decoder_layers: int = 7
    cvae_encoder_layers: int = 4
    heads: int = 8
    feedforward_dim: int = 3200
    dropout: float = 0.1


@dataclass(frozen=True)
class ACTTrainingConfig:
    """Configuration for ACT losses and temporal ensembling."""

    beta: float = 10.0
    reconstruction_loss: str = "l1"
    temporal_ensemble_decay: float = 0.01


@dataclass(frozen=True)
class ACTConfig:
    """Top-level configuration for the educational ACT model."""

    observation_config: ACTObservationConfig = field(default_factory=ACTObservationConfig)
    vision_config: ACTVisionConfig = field(default_factory=ACTVisionConfig)
    transformer_config: ACTTransformerConfig = field(default_factory=ACTTransformerConfig)
    training_config: ACTTrainingConfig = field(default_factory=ACTTrainingConfig)


@dataclass
class ACTOutput:
    """Container returned by ``ACTModel.forward``.

    Attributes:
        visual_tokens: Multi-camera visual tokens with shape ``[B, V, D]``.
        encoder_context: Fused context sequence used as decoder memory.
        latent_mean: Optional CVAE posterior mean.
        latent_logvar: Optional CVAE posterior log-variance.
        latent_sample: Latent style variable used by the decoder.
        action_sequence: Predicted action chunk with shape ``[B, K, A]``.
        reconstruction_loss: Optional action reconstruction loss.
        kl_loss: Optional CVAE KL loss.
        loss: Optional total loss.
    """

    visual_tokens: Tensor
    encoder_context: Tensor
    latent_mean: Tensor | None
    latent_logvar: Tensor | None
    latent_sample: Tensor
    action_sequence: Tensor
    reconstruction_loss: Tensor | None = None
    kl_loss: Tensor | None = None
    loss: Tensor | None = None


class ACTImageTokenizer(nn.Module):
    """Compact multi-camera CNN tokenizer.

    The paper uses one ResNet18 per camera and obtains ``15 x 20`` feature grids
    for ``480 x 640`` images. This educational version uses a smaller shared CNN
    followed by adaptive pooling to keep the same tokenization interface.
    """

    def __init__(self, observation_config: ACTObservationConfig, vision_config: ACTVisionConfig) -> None:
        super().__init__()
        self.observation_config = observation_config
        self.vision_config = vision_config

        self.cnn = nn.Sequential(
            nn.Conv2d(observation_config.in_channels, vision_config.stem_width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(vision_config.stem_width, vision_config.stem_width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(vision_config.stem_width, vision_config.visual_width, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(vision_config.dropout)
        self.camera_embedding = nn.Embedding(
            observation_config.num_cameras,
            vision_config.visual_width,
        )
        position_embedding = _build_2d_sincos_position_embedding(
            observation_config.feature_grid_height,
            observation_config.feature_grid_width,
            vision_config.visual_width,
        )
        self.register_buffer("position_embedding", position_embedding, persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        """Tokenizes multi-camera images.

        Args:
            images: Tensor with shape ``[B, Cams, C, H, W]``.

        Returns:
            Visual tokens with shape ``[B, Cams * grid_h * grid_w, visual_width]``.
        """

        expected_cameras = self.observation_config.num_cameras
        if images.dim() != 5:
            raise ValueError("images must have shape [B, Cams, C, H, W].")
        if images.size(1) != expected_cameras:
            raise ValueError(f"Expected {expected_cameras} cameras, got {images.size(1)}.")

        batch_size, num_cameras, channels, height, width = images.shape
        images = images.reshape(batch_size * num_cameras, channels, height, width)
        features = self.cnn(images)
        features = F.adaptive_avg_pool2d(
            features,
            (
                self.observation_config.feature_grid_height,
                self.observation_config.feature_grid_width,
            ),
        )
        features = features.flatten(2).transpose(1, 2)
        features = features + self.position_embedding.unsqueeze(0).to(features.dtype)
        features = features.reshape(
            batch_size,
            num_cameras,
            self.observation_config.tokens_per_camera,
            self.vision_config.visual_width,
        )

        camera_ids = torch.arange(num_cameras, device=features.device)
        camera_tokens = self.camera_embedding(camera_ids).view(1, num_cameras, 1, -1)
        features = features + camera_tokens
        features = features.reshape(
            batch_size,
            self.observation_config.total_visual_tokens,
            self.vision_config.visual_width,
        )
        return self.dropout(features)


class ACTStyleEncoder(nn.Module):
    """CVAE encoder that infers the latent style variable from action chunks."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        observation_config = config.observation_config
        transformer_config = config.transformer_config
        hidden_dim = transformer_config.hidden_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.joint_projection = nn.Linear(observation_config.joint_dim, hidden_dim)
        self.action_projection = nn.Linear(observation_config.action_dim, hidden_dim)
        action_position_embedding = _build_1d_sincos_position_embedding(
            transformer_config.chunk_size,
            hidden_dim,
        )
        self.register_buffer("action_position_embedding", action_position_embedding, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_config.heads,
            dim_feedforward=transformer_config.feedforward_dim,
            dropout=transformer_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config.cvae_encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.mean_head = nn.Linear(hidden_dim, transformer_config.latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, transformer_config.latent_dim)

    def forward(self, joints: Tensor, target_action_sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Infers posterior parameters for the CVAE style variable."""

        batch_size, chunk_size, _ = target_action_sequence.shape
        if chunk_size > self.action_position_embedding.size(0):
            raise ValueError("target_action_sequence is longer than configured chunk_size.")

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        joint_token = self.joint_projection(joints).unsqueeze(1)
        action_tokens = self.action_projection(target_action_sequence)
        action_tokens = action_tokens + self.action_position_embedding[:chunk_size].to(action_tokens.dtype)
        encoder_input = torch.cat([cls_token, joint_token, action_tokens], dim=1)
        encoded = self.encoder(encoder_input)
        cls_hidden = encoded[:, 0]
        return self.mean_head(cls_hidden), self.logvar_head(cls_hidden)


class ACTPolicyDecoder(nn.Module):
    """ACT decoder policy that maps observations and style to an action chunk."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        observation_config = config.observation_config
        vision_config = config.vision_config
        transformer_config = config.transformer_config
        hidden_dim = transformer_config.hidden_dim

        self.config = config
        self.image_tokenizer = ACTImageTokenizer(observation_config, vision_config)
        self.visual_projection = nn.Linear(vision_config.visual_width, hidden_dim)
        self.joint_projection = nn.Linear(observation_config.joint_dim, hidden_dim)
        self.latent_projection = nn.Linear(transformer_config.latent_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_config.heads,
            dim_feedforward=transformer_config.feedforward_dim,
            dropout=transformer_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config.encoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=transformer_config.heads,
            dim_feedforward=transformer_config.feedforward_dim,
            dropout=transformer_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=transformer_config.decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        query_embedding = _build_1d_sincos_position_embedding(
            transformer_config.chunk_size,
            hidden_dim,
        )
        self.register_buffer("query_embedding", query_embedding, persistent=False)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, observation_config.action_dim),
        )

    def forward(self, images: Tensor, joints: Tensor, latent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predicts a future action sequence from the current observation."""

        batch_size = images.size(0)
        visual_tokens = self.image_tokenizer(images)
        visual_tokens = self.visual_projection(visual_tokens)
        joint_token = self.joint_projection(joints).unsqueeze(1)
        latent_token = self.latent_projection(latent).unsqueeze(1)

        context_tokens = torch.cat([visual_tokens, joint_token, latent_token], dim=1)
        encoder_context = self.context_encoder(context_tokens)

        action_queries = self.query_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        action_hidden = self.action_decoder(action_queries, encoder_context)
        action_sequence = self.action_head(action_hidden)
        return visual_tokens, encoder_context, action_sequence


class ACTTemporalEnsembler:
    """Buffers and averages overlapping ACT action chunks.

    This helper mirrors Algorithm 2 in the paper at the interface level. It is
    meant for study and offline rollouts, not as a hard real-time controller.
    """

    def __init__(self, *, decay: float, max_timesteps: int = 10_000) -> None:
        self.decay = decay
        self.max_timesteps = max_timesteps
        self._buffers: list[list[tuple[int, Tensor]]] = [[] for _ in range(max_timesteps)]

    def reset(self) -> None:
        """Clears all stored chunk predictions."""

        self._buffers = [[] for _ in range(self.max_timesteps)]

    def add_chunk(self, start_timestep: int, action_chunk: Tensor) -> None:
        """Adds one predicted action chunk to future timestep buffers.

        Args:
            start_timestep: Timestep at which the chunk was predicted.
            action_chunk: Tensor with shape ``[B, K, A]``.
        """

        if action_chunk.dim() != 3:
            raise ValueError("action_chunk must have shape [B, K, A].")

        chunk_size = action_chunk.size(1)
        for offset in range(chunk_size):
            target_timestep = start_timestep + offset
            if target_timestep >= self.max_timesteps:
                break
            self._buffers[target_timestep].append((start_timestep, action_chunk[:, offset].detach()))

    def get_action(self, timestep: int) -> Tensor:
        """Returns the temporally ensembled action for a timestep."""

        if timestep >= self.max_timesteps:
            raise ValueError("timestep exceeds max_timesteps.")
        predictions = self._buffers[timestep]
        if not predictions:
            raise ValueError(f"No action predictions stored for timestep={timestep}.")

        weighted_action = None
        total_weight = 0.0
        for source_timestep, action in predictions:
            prediction_age = timestep - source_timestep
            weight = math.exp(-self.decay * prediction_age)
            total_weight += weight
            contribution = action * weight
            weighted_action = contribution if weighted_action is None else weighted_action + contribution

        if weighted_action is None:
            raise RuntimeError("Temporal ensemble buffer unexpectedly empty.")
        return weighted_action / total_weight


class ACTModel(nn.Module):
    """Educational ACT policy with CVAE training and chunked inference."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.config = config
        self.style_encoder = ACTStyleEncoder(config)
        self.policy = ACTPolicyDecoder(config)

    def _sample_latent(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Samples a latent style variable with the reparameterization trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def _zero_latent(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Returns the deterministic inference latent used by ACT."""

        return torch.zeros(
            batch_size,
            self.config.transformer_config.latent_dim,
            device=device,
            dtype=dtype,
        )

    def _reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Computes the configured action reconstruction loss."""

        loss_type = self.config.training_config.reconstruction_loss
        if loss_type == "l1":
            return F.l1_loss(prediction, target)
        if loss_type == "mse":
            return F.mse_loss(prediction, target)
        raise ValueError("reconstruction_loss must be either 'l1' or 'mse'.")

    def _kl_loss(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Computes KL(q(z|x) || N(0, I)) averaged over the batch."""

        per_sample = -0.5 * torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        return per_sample.mean()

    def forward(
        self,
        images: Tensor,
        joints: Tensor,
        *,
        target_action_sequence: Tensor | None = None,
        latent: Tensor | None = None,
    ) -> ACTOutput:
        """Runs training or inference-style ACT forward pass.

        Args:
            images: Multi-camera observations with shape ``[B, Cams, C, H, W]``.
            joints: Current follower joint positions with shape ``[B, joint_dim]``.
            target_action_sequence: Optional training target with shape
                ``[B, chunk_size, action_dim]``.
            latent: Optional explicit style variable. If omitted during
                inference, ACT uses the zero latent.

        Returns:
            ``ACTOutput`` containing predicted action chunks and optional losses.
        """

        latent_mean = None
        latent_logvar = None
        reconstruction_loss = None
        kl_loss = None
        total_loss = None

        if target_action_sequence is not None:
            latent_mean, latent_logvar = self.style_encoder(joints, target_action_sequence)
            latent_sample = self._sample_latent(latent_mean, latent_logvar)
        elif latent is not None:
            latent_sample = latent
        else:
            latent_sample = self._zero_latent(
                batch_size=images.size(0),
                device=images.device,
                dtype=images.dtype,
            )

        visual_tokens, encoder_context, action_sequence = self.policy(images, joints, latent_sample)

        if target_action_sequence is not None:
            if target_action_sequence.shape != action_sequence.shape:
                raise ValueError(
                    "target_action_sequence must match predicted action_sequence shape; "
                    f"got target={tuple(target_action_sequence.shape)} and "
                    f"prediction={tuple(action_sequence.shape)}."
                )
            reconstruction_loss = self._reconstruction_loss(action_sequence, target_action_sequence)
            if latent_mean is None or latent_logvar is None:
                raise RuntimeError("CVAE posterior was not computed for training forward.")
            kl_loss = self._kl_loss(latent_mean, latent_logvar)
            total_loss = reconstruction_loss + self.config.training_config.beta * kl_loss

        return ACTOutput(
            visual_tokens=visual_tokens,
            encoder_context=encoder_context,
            latent_mean=latent_mean,
            latent_logvar=latent_logvar,
            latent_sample=latent_sample,
            action_sequence=action_sequence,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            loss=total_loss,
        )

    @torch.no_grad()
    def predict_action_chunk(self, images: Tensor, joints: Tensor) -> Tensor:
        """Predicts a deterministic future action chunk with ``z = 0``."""

        output = self(images, joints)
        return output.action_sequence

    @torch.no_grad()
    def predict_ensembled_action(
        self,
        images: Tensor,
        joints: Tensor,
        *,
        timestep: int,
        ensembler: ACTTemporalEnsembler,
    ) -> Tensor:
        """Predicts a chunk, updates the ensemble buffer, and returns action ``a_t``."""

        action_chunk = self.predict_action_chunk(images, joints)
        ensembler.add_chunk(timestep, action_chunk)
        return ensembler.get_action(timestep)

    def create_temporal_ensembler(self, *, max_timesteps: int = 10_000) -> ACTTemporalEnsembler:
        """Creates a temporal ensemble buffer using the model's decay setting."""

        return ACTTemporalEnsembler(
            decay=self.config.training_config.temporal_ensemble_decay,
            max_timesteps=max_timesteps,
        )


def build_act_tiny() -> ACTModel:
    """Builds a tiny ACT model for smoke tests and study."""

    config = ACTConfig(
        observation_config=ACTObservationConfig(
            num_cameras=4,
            image_height=64,
            image_width=64,
            joint_dim=14,
            action_dim=14,
            feature_grid_height=4,
            feature_grid_width=4,
        ),
        vision_config=ACTVisionConfig(
            stem_width=32,
            visual_width=64,
        ),
        transformer_config=ACTTransformerConfig(
            hidden_dim=64,
            latent_dim=16,
            chunk_size=8,
            encoder_layers=2,
            decoder_layers=2,
            cvae_encoder_layers=2,
            heads=4,
            feedforward_dim=256,
            dropout=0.0,
        ),
        training_config=ACTTrainingConfig(
            beta=0.1,
            reconstruction_loss="l1",
            temporal_ensemble_decay=0.1,
        ),
    )
    return ACTModel(config)


def _smoke_test() -> None:
    """Runs a tiny training and inference check."""

    torch.manual_seed(0)
    model = build_act_tiny()
    model.eval()

    batch_size = 2
    config = model.config
    images = torch.randn(
        batch_size,
        config.observation_config.num_cameras,
        config.observation_config.in_channels,
        config.observation_config.image_height,
        config.observation_config.image_width,
    )
    joints = torch.randn(batch_size, config.observation_config.joint_dim)
    target_actions = torch.randn(
        batch_size,
        config.transformer_config.chunk_size,
        config.observation_config.action_dim,
    )

    output = model(images, joints, target_action_sequence=target_actions)
    action_chunk = model.predict_action_chunk(images, joints)
    ensembler = model.create_temporal_ensembler(max_timesteps=32)
    action_t0 = model.predict_ensembled_action(
        images,
        joints,
        timestep=0,
        ensembler=ensembler,
    )
    model.predict_ensembled_action(images, joints, timestep=1, ensembler=ensembler)
    action_t1 = ensembler.get_action(1)

    print(f"visual_tokens: {tuple(output.visual_tokens.shape)}")
    print(f"encoder_context: {tuple(output.encoder_context.shape)}")
    print(f"latent_mean: {tuple(output.latent_mean.shape) if output.latent_mean is not None else None}")
    print(f"action_sequence: {tuple(output.action_sequence.shape)}")
    print(f"loss: {float(output.loss.detach()):.4f}")
    print(f"predicted_chunk: {tuple(action_chunk.shape)}")
    print(f"ensembled_action_t0: {tuple(action_t0.shape)}")
    print(f"ensembled_action_t1: {tuple(action_t1.shape)}")


if __name__ == "__main__":
    _smoke_test()
