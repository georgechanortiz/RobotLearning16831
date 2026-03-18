# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRLEnv subclass with bounded action space for off-policy algorithms (e.g. SAC).

ManagerBasedRLEnv exposes (-inf, inf) action bounds by default, which causes
NaN samples during the random exploration phase of off-policy RL.  This
subclass overrides ``action_space`` to return finite [-1, 1] bounds (matching
DirectRLEnv behaviour), so that ``env.action_space.sample()`` always produces
valid actions.

See: https://github.com/isaac-sim/IsaacLab/issues/3064
"""

import gymnasium as gym
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv


class BoundedManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a finite [-1, 1] action space."""

    @property
    def action_space(self) -> gym.spaces.Box:
        base_space = self.action_manager.action_space
        return gym.spaces.Box(
            low=np.full(base_space.shape, -1.0, dtype=np.float32),
            high=np.full(base_space.shape, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
