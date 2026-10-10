"""Week 3 VLA training-engineering smoke test.

Synthetic, episode-aware robot data -> train-only normalization -> action chunks
-> valid masks -> conditional flow-matching loss -> optimizer/checkpoint.

Run: python week3_minimal_flow_train.py
Requires: torch
This is an EDUCATIONAL action-policy skeleton, not a faithful pi0/LeRobot replica.
"""

from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


SEED = 17
ACTION_HORIZON = 12
OBS_DIM = 12  # Already-encoded observation features: NOT raw RGB / VLM tokens.
STATE_DIM = 6
ACTION_DIM = 7


@dataclass
class Episode:
    observation: torch.Tensor  # [T, OBS_DIM]
    state: torch.Tensor        # [T, STATE_DIM]
    action: torch.Tensor       # [T, ACTION_DIM], command issued at this t


def make_episodes(n: int = 18) -> list[Episode]:
    """Smooth toy signals. Every episode is an independent trajectory."""
    g = torch.Generator().manual_seed(SEED)
    eps = []
    for _ in range(n):
        T = int(torch.randint(18, 55, (1,), generator=g).item())
        # Smooth features -> predictable trajectories (not a robot simulator).
        increments = torch.randn(T, OBS_DIM, generator=g) * 0.09
        obs = increments.cumsum(dim=0) + torch.randn(1, OBS_DIM, generator=g) * 0.3
        state = obs[:, :STATE_DIM] + torch.randn(T, STATE_DIM, generator=g) * 0.025
        action = torch.cat(
            [0.4 * state + 0.15 * obs[:, 6:12],
             torch.sigmoid(3 * obs[:, 0:1])], dim=-1
        ) + torch.randn(T, ACTION_DIM, generator=g) * 0.025
        eps.append(Episode(obs.float(), state.float(), action.float()))
    return eps


class ChunkDataset(Dataset):
    """Never crosses episode boundaries; last windows are right-padded and masked."""

    def __init__(self, episodes: list[Episode], action_mean, action_std, horizon: int):
        self.episodes = episodes
        self.mean = action_mean
        self.std = action_std.clamp_min(1e-6)
        self.horizon = horizon
        self.indices = [(ep_id, t) for ep_id, e in enumerate(episodes)
                        for t in range(e.action.shape[0])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_id, t = self.indices[idx]
        episode = self.episodes[ep_id]
        future = episode.action[t:t + self.horizon]
        valid_length = len(future)
        padded = torch.zeros(self.horizon, ACTION_DIM, dtype=torch.float32)
        # Normalize before padding so padded values are guaranteed to be zeros.
        padded[:valid_length] = (future - self.mean) / self.std
        mask = torch.zeros(self.horizon, 1, dtype=torch.float32)
        mask[:valid_length] = 1.0
        return {
            'observation': episode.observation[t],
            'state': episode.state[t],
            'actions': padded,
            'action_mask': mask,
            'episode_id': ep_id,
            'frame_id': t,
        }


class TinyFlowExpert(nn.Module):
    """Small MLP action expert conditioned on observation, state and flow time."""

    def __init__(self):
        super().__init__()
        input_dim = OBS_DIM + STATE_DIM + ACTION_DIM + 2  # flow t + chunk position
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, observation, state, x_t, flow_t):
        B, H, _ = x_t.shape
        context = torch.cat([observation, state], dim=-1)
        context = context[:, None, :].expand(-1, H, -1)
        tau = flow_t.expand(B, H, 1)
        position = torch.linspace(0, 1, H, device=x_t.device, dtype=x_t.dtype)
        position = position.view(1, H, 1).expand(B, -1, -1)
        return self.network(torch.cat([context, x_t, tau, position], dim=-1))


def flow_loss(model, batch, device):
    a = batch['actions'].to(device)
    c = batch['observation'].to(device)
    s = batch['state'].to(device)
    valid = batch['action_mask'].to(device).expand_as(a)
    B = a.shape[0]
    tau = torch.rand(B, 1, 1, device=device)
    noise = torch.randn_like(a)
    x_t = (1 - tau) * noise + tau * a
    target = a - noise
    prediction = model(c, s, x_t, tau)
    # Pad positions are masked from the objective, including their velocity target.
    sq_error = (prediction - target).float().square()
    return (sq_error * valid).sum() / valid.sum().clamp_min(1.0)


def smoke_tests(train_ds):
    # Must not mistakenly draw the next episode's actions into a chunk.
    ep_id = 0
    last_t = len(train_ds.episodes[ep_id].action) - 1
    idx = train_ds.indices.index((ep_id, last_t))
    s = train_ds[idx]
    assert s['actions'].shape == (ACTION_HORIZON, ACTION_DIM)
    assert s['action_mask'].shape == (ACTION_HORIZON, 1)
    assert int(s['action_mask'].sum()) == 1, 'Episode boundary leaked'
    assert torch.equal(s['actions'][1:], torch.zeros_like(s['actions'][1:]))
    assert bool(torch.isfinite(s['actions']).all())


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    episodes = make_episodes()

    # Split by EPISODE before fitting normalization, never by random frames.
    rng = random.Random(SEED)
    ids = list(range(len(episodes)))
    rng.shuffle(ids)
    cut = int(len(ids) * 0.8)
    train_eps = [episodes[i] for i in ids[:cut]]
    valid_eps = [episodes[i] for i in ids[cut:]]

    # Training statistics only, computed on actual action frames (not padded chunks).
    raw_train_actions = torch.cat([ep.action for ep in train_eps], dim=0)
    mean = raw_train_actions.mean(dim=0)
    std = raw_train_actions.std(dim=0).clamp_min(1e-6)
    train_ds = ChunkDataset(train_eps, mean, std, ACTION_HORIZON)
    valid_ds = ChunkDataset(valid_eps, mean, std, ACTION_HORIZON)
    smoke_tests(train_ds)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)
    model = TinyFlowExpert().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # CPU and CUDA compatible: CUDA autocast is optional; no dependency on GradScaler.
    for epoch in range(3):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda' and torch.cuda.is_bf16_supported())):
                loss = flow_loss(model, batch, device)
            assert torch.isfinite(loss), 'Loss diverged'
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = [flow_loss(model, batch, device).item() for batch in valid_loader]
        print(f'epoch={epoch + 1} train_loss={sum(train_losses)/len(train_losses):.4f}'
              f' valid_loss={sum(val_losses)/len(val_losses):.4f}')

    output = Path(__file__).with_name('week3_toy_checkpoint.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'train_action_mean': mean,
        'train_action_std': std,
        'action_horizon': ACTION_HORIZON,
        'note': 'Only for Week 3 educational smoke tests',
    }, output)
    print(f'Smoke tests passed; checkpoint saved: {output}')


if __name__ == '__main__':
    main()
