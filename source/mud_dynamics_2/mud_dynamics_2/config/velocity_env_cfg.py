# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Pre-defined configs
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import torch
import isaaclab.utils.math as math_utils
# from forces_cfg import suction_and_resistive_forces

# Scene definition
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )



    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )





# MDP settings
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), 
            lin_vel_y=(0.0, 0.0), 
            ang_vel_z=(-0.5, 0.5), heading=(-math.pi, math.pi)
            # lin_vel_x=(-1.0, 1.0), 
            # lin_vel_y=(-1.0, 1.0), 
            # ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-1.0, 1.0))



        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

def suction_and_resistive_forces(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, c_range: tuple[float, float], z_threshold: float):
    asset: ArticulationCfg = env.scene[asset_cfg.name]
    root_pos = asset.data.body_link_pos_w[env_ids]    # root_pos: [num_envs, num_bodies, 3]
    root_vel = asset.data.body_link_vel_w[env_ids]    # root_vel: [num_envs, num_bodies, 3]

    z_pos = root_pos[..., 2] # Extract only the Z positions
    z_vel = root_vel[..., 2] # Extract only the Z velocities

    in_mud = z_pos < z_threshold    # Bodies inside mud (Bool)
    depth_ratio = (((z_threshold - z_pos) / z_threshold).clamp(min=0.0)) ** 2    # Quadratic depth scaling (clamped for safety)
    coeffs = torch.empty_like(z_pos).uniform_(*c_range)     # Randomize the coefficients - look into if uniform dist is what i want 

    force_sign = torch.where(z_vel > 0, -1.0, 1.0) # Force direction: + vel > suction > - force  and  - vel > resistive > + force
    z_force = coeffs * depth_ratio * force_sign * in_mud   # Final Z force

    # Construct force tensor
    forces = torch.zeros((*z_force.shape, 1, 3), device=asset.device)    # [num_envs, num_bodies, 1, 3]
    forces[..., 0, 2] = z_force 
    torques = torch.zeros_like(forces)

    # print("suction and resistive force",forces)

    # Apply per environment
    for i, env_id in enumerate(env_ids):
        body_ids = torch.nonzero(in_mud[i], as_tuple=False).squeeze(-1)
        if body_ids.numel() == 0:
            continue
        asset.instantaneous_wrench_composer.set_forces_and_torques(forces=forces[i, body_ids], torques=torques[i, body_ids], env_ids=env_id.unsqueeze(0), body_ids=body_ids) # Set and write forces 
    asset.write_data_to_sim()


def shear_forces(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, c1: tuple[float, float], c2: tuple[float, float], z_threshold: float, area: float):
    asset: ArticulationCfg = env.scene[asset_cfg.name]
    root_pos = asset.data.body_link_pos_w[env_ids]    # root_pos: [num_envs, num_bodies, 3]
    root_vel = asset.data.body_link_vel_w[env_ids]    # root_vel: [num_envs, num_bodies, 3]
    
    # Extract positions and velocities
    z_pos = root_pos[..., 2]   
    xy_pos = root_pos[..., 0:2]   
    xy_vel = root_vel[..., 0:2]    
    
    # Calculate magnitudes
    pos_magnitude = torch.linalg.norm(xy_pos, dim=-1, keepdim=True)
    vel_magnitude = torch.linalg.norm(xy_vel, dim=-1, keepdim=True)
    
    # Angle of velocity vector in XY plane
    angle = torch.atan2(xy_vel[..., 1], xy_vel[..., 0]) + torch.pi
    
    in_mud = z_pos < z_threshold    # Bodies inside mud (Bool)
    
    # Shear rate and stress calculation
    shear_rate = vel_magnitude / (pos_magnitude + 1e-6)  # Add epsilon to avoid division by zero
    
    # Randomize coefficients
    m = torch.empty(1, device=asset.device).uniform_(*c1)
    b = torch.empty(1, device=asset.device).uniform_(*c2)
    
    shear_stress = area * torch.log(shear_rate + 1e-3) + m * shear_rate + b
    
    # Calculate shear forces for all bodies
    xy_force = shear_stress * in_mud.unsqueeze(-1)  # [num_envs, num_bodies, 1]
    x_force = xy_force.squeeze(-1) * torch.cos(angle)  # [num_envs, num_bodies]
    y_force = xy_force.squeeze(-1) * torch.sin(angle)  # [num_envs, num_bodies]
    
    # Construct force tensor
    forces = torch.zeros((*z_pos.shape, 1, 3), device=asset.device)    # [num_envs, num_bodies, 1, 3]
    forces[..., 0, 0] = x_force  # X component for all bodies
    forces[..., 0, 1] = y_force  # Y component for all bodies
    
    torques = torch.zeros_like(forces)

    # print("shear force",forces)
    print
    # Apply per environment
    for i, env_id in enumerate(env_ids):
        body_ids = torch.nonzero(in_mud[i], as_tuple=False).squeeze(-1)
        if body_ids.numel() == 0:
            continue
        asset.instantaneous_wrench_composer.set_forces_and_torques(forces=forces[i, body_ids], torques=torques[i, body_ids], env_ids=env_id.unsqueeze(0), body_ids=body_ids)
    asset.write_data_to_sim()




@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # forces
    suction_and_resistive_force_term = EventTerm(
        func=suction_and_resistive_forces,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "c_range": (14.0, 15.0),
            "z_threshold": 0.1143,
        },
    )

    shear_force_term = EventTerm(
        func=shear_forces,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "c1": (4.0,10.0),
            "c2": (6.0, 18.0),
            "z_threshold": 0.1143,
            "area": 0.2
        },
    )
    
    # resistance_force_term = EventTerm(
    #     func=apply_resistive_force,
    #     mode="interval",
    #     interval_range_s=(0.1, 0.1),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "c_range": (940.0, 950.0),
    #         "z_threshold": 0.15,
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=0.25, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "command_name": "base_velocity", "threshold": 2.5})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0})
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

##
# Environment configuration
##

@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
