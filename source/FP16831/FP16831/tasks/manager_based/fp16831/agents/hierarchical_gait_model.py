# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom skrl models for hierarchical gait-conditioned locomotion policies."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils.runner.torch import Runner


class HierarchicalGaitPolicy(GaussianMixin, Model):
    """Gaussian joint-command policy with an explicit high-level gait selector.

    The gait selector produces a distribution over learned gait modes. During
    training, the low-level controller receives the soft gait probabilities so
    PPO can still optimize a standard continuous Gaussian action policy.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cuda:0",
        num_gaits: int = 4,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: type[nn.Module] = nn.ELU,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        if num_gaits < 2:
            raise ValueError("num_gaits must be at least 2")

        self.num_gaits = num_gaits
        encoder_layers: list[nn.Module] = []
        in_dim = self.num_observations
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden_dim), activation()])
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.gait_head = nn.Linear(in_dim, num_gaits)
        self.low_level = nn.Sequential(
            nn.Linear(in_dim + num_gaits, hidden_dims[-1]),
            activation(),
            nn.Linear(hidden_dims[-1], self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))
        self.to(self.device)

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = ""):
        states = inputs["states"]
        latent = self.encoder(states)
        gait_logits = self.gait_head(latent)
        gait_probs = torch.softmax(gait_logits, dim=-1)
        mean_actions = self.low_level(torch.cat([latent, gait_probs], dim=-1))
        return mean_actions, self.log_std_parameter, {
            "gait_logits": gait_logits,
            "gait_probs": gait_probs,
            "gait_id": torch.argmax(gait_probs, dim=-1),
        }


class GaitValueModel(DeterministicMixin, Model):
    """State-value network paired with :class:`HierarchicalGaitPolicy`."""

    def __init__(
        self,
        observation_space,
        action_space,
        device: str | torch.device = "cuda:0",
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: type[nn.Module] = nn.ELU,
        clip_actions: bool = False,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        layers: list[nn.Module] = []
        in_dim = self.num_observations
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), activation()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.to(self.device)

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = ""):
        return self.net(inputs["states"]), {}


class HierarchicalGaitRunner:
    """Small runner facade matching the skrl Runner methods used by scripts."""

    def __init__(self, agent: PPO, trainer: SequentialTrainer) -> None:
        self.agent = agent
        self.trainer = trainer

    def run(self, mode: str = "train") -> None:
        if mode == "train":
            self.trainer.train()
        elif mode == "eval":
            self.trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")


def _process_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Use skrl's config conversion so YAML strings match Runner behavior."""

    cfg = copy.deepcopy(cfg)
    cfg["rewards_shaper"] = cfg.get("rewards_shaper")
    runner = object.__new__(Runner)
    return runner._process_cfg(cfg)


def build_hierarchical_gait_runner(env, cfg: dict[str, Any]) -> HierarchicalGaitRunner:
    """Construct a PPO runner using the custom hierarchical gait policy."""

    if cfg.get("custom_model") != "hierarchical_gait":
        raise ValueError("build_hierarchical_gait_runner requires custom_model: hierarchical_gait")

    device = env.device
    observation_space = env.observation_space
    action_space = env.action_space

    model_cfg = cfg.get("hierarchical_gait", {})
    hidden_dims = tuple(model_cfg.get("hidden_dims", [128, 128, 128]))
    num_gaits = int(model_cfg.get("num_gaits", 4))
    initial_log_std = float(model_cfg.get("initial_log_std", 0.0))

    models = {
        "policy": HierarchicalGaitPolicy(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            num_gaits=num_gaits,
            hidden_dims=hidden_dims,
            initial_log_std=initial_log_std,
        ),
        "value": GaitValueModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            hidden_dims=hidden_dims,
        ),
    }

    memory_cfg = copy.deepcopy(cfg.get("memory", {"class": "RandomMemory", "memory_size": -1}))
    memory_cfg.pop("class", None)
    if memory_cfg.get("memory_size", -1) < 0:
        memory_cfg["memory_size"] = cfg["agent"]["rollouts"]
    memory = RandomMemory(num_envs=env.num_envs, device=device, **memory_cfg)

    agent_cfg = copy.deepcopy(PPO_DEFAULT_CONFIG)
    agent_cfg.update(_process_cfg(cfg["agent"]))
    agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": observation_space, "device": device})
    agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})

    agent = PPO(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        cfg=agent_cfg,
    )

    trainer_cfg = copy.deepcopy(cfg["trainer"])
    trainer_cfg.pop("class", None)
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)
    return HierarchicalGaitRunner(agent=agent, trainer=trainer)
