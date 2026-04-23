# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RandFlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_slide.weight = -0.08
        self.rewards.foot_contact_impact.weight = -2.0e-5
        self.rewards.contact_mode_change.weight = -0.03
        self.rewards.action_rate_l2.weight = -0.03

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # disable startup randomization terms that are incompatible with the installed Isaac Lab event API
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None




class UnitreeGo2RandFlatEnvCfg_PLAY(UnitreeGo2RandFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
