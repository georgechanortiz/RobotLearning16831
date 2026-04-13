from __future__ import annotations

import torch

from .models import DynamicsEnsemble


class CEMPlanner:
    """Cross-entropy planner over learned dynamics."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 12,
        candidates: int = 256,
        elites: int = 32,
        iterations: int = 4,
        discount: float = 0.99,
        temperature: float = 0.5,
    ):
        self.model = model
        self.action_low = action_low
        self.action_high = action_high
        self.horizon = horizon
        self.candidates = candidates
        self.elites = elites
        self.iterations = iterations
        self.discount = discount
        self.temperature = temperature
        self.action_dim = action_low.numel()

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        mean = torch.zeros(
            (batch_size, self.horizon, self.action_dim), device=obs.device, dtype=obs.dtype
        )
        std = torch.full_like(mean, self.temperature)

        for _ in range(self.iterations):
            noise = torch.randn(
                (batch_size, self.candidates, self.horizon, self.action_dim), device=obs.device, dtype=obs.dtype
            )
            action_sequences = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            action_sequences = torch.max(
                torch.min(action_sequences, self.action_high.view(1, 1, 1, -1)),
                self.action_low.view(1, 1, 1, -1),
            )

            returns = self.evaluate_sequences(obs, action_sequences)
            elite_indices = returns.topk(self.elites, dim=1).indices
            elite_actions = action_sequences.gather(
                1, elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim)
            )
            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1).clamp_min(1e-3)

        return mean[:, 0, :]

    @torch.no_grad()
    def evaluate_sequences(self, obs: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        batch_size, candidates, _, action_dim = action_sequences.shape
        states = obs.unsqueeze(1).expand(-1, candidates, -1).reshape(batch_size * candidates, -1)
        returns = torch.zeros(batch_size * candidates, device=obs.device, dtype=obs.dtype)
        discounts = torch.ones_like(returns)
        alive = torch.ones_like(returns)

        for t in range(self.horizon):
            actions_t = action_sequences[:, :, t, :].reshape(batch_size * candidates, action_dim)
            preds = self.model.predict(states, actions_t)
            reward = preds.rewards.squeeze(-1)
            continue_prob = preds.continue_logits.sigmoid().squeeze(-1)
            returns = returns + discounts * alive * reward
            alive = alive * continue_prob
            discounts = discounts * self.discount
            states = states + preds.delta_obs

        return returns.view(batch_size, candidates)
