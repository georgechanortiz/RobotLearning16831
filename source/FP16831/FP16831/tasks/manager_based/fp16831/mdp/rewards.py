# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def foot_contact_impact_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 100.0
) -> torch.Tensor:
    """Penalize high-impact foot contacts above a force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak_forces = torch.linalg.norm(contact_forces, dim=-1).max(dim=1).values
    excess_forces = torch.clamp(peak_forces - threshold, min=0.0)
    return torch.sum(torch.square(excess_forces), dim=1)


def contact_mode_change_l1(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize abrupt changes in the foot contact pattern.

    This is a reward-level proxy for unnecessary gait switching. It discourages
    contact pattern churn while still allowing transitions when task rewards
    make them beneficial.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    if contact_history.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)

    current_contacts = torch.linalg.norm(contact_history[:, 0], dim=-1) > threshold
    previous_contacts = torch.linalg.norm(contact_history[:, 1], dim=-1) > threshold
    return torch.sum(current_contacts != previous_contacts, dim=1).float()
