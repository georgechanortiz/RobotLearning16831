'''
    # Usage
    python scripts/my_custom_mud_environment.py --num_envs 1
'''
import argparse


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--agent", type=str, default=None, help=( "Name of the RL agent configuration entry point. Defaults to None, in which case the argument ""--algorithm is used to determine the default agent configuration entry point."),)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
#app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
import isaaclab.utils.math as math
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Pre-defined configs
from mud_dynamics_2.robots.go2 import UNITREE_GO2_CFG

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import RigidPrim
# from omni.physics.tensors import RigidContactView
import omni.physics.tensors as tensors
# from omni.physics.tensors import utils
from omni.physx.scripts.utils import addPairFilter

from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
import omni.physx as physx


import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os



CUSTOM_GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2,                # coefficient of static friction
    dynamic_friction=1.0,               # coefficient of dynamic friction
    restitution=0.0,                    # bounciness
    friction_combine_mode="multiply",   # penalty stiffness (contact stiffness)
    restitution_combine_mode="average",
    compliant_contact_stiffness=1e5,    # compliant stiffness
    compliant_contact_damping=1e3,      # compliant damping
)

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for a Go2 scene."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(physics_material=CUSTOM_GROUND_MATERIAL))# ground plane
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))        # lights
    Go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",)                                                    # articulation


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def run_simulator(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict): # (sim: SimulationContext, scene: InteractiveScene):#

    if args_cli.agent is None:
        algorithm = args_cli.algorithm.lower()
        agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()



    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])           # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")


    if args_cli.checkpoint is None:
        print("[ERROR]: Please provide a valid checkpoint path using --checkpoint argument.")
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model from checkpoint: {resume_path}")
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])


    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array") 

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.01
    sim_cfg.gravity = (0.0, 0.0, -9.81)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([10, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = Go2SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    robot_positions = run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()












# '''
#     # Usage
#     python scripts/my_custom_mud_environment.py --num_envs 1
# '''

# """Launch Isaac Sim Simulator first."""
# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
# parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")


# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""



# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.sim import SimulationContext
# from isaaclab.utils import configclass
# import isaaclab.utils.math as math



# # Pre-defined configs
# from mud_dynamics_2.robots.go2 import UNITREE_GO2_CFG

# from isaacsim.core.utils.stage import get_current_stage
# from isaacsim.core.prims import RigidPrim
# # from omni.kit.usd
# physics.tensors import RigidContactView
# import omni.physics.tensors as tensors
# # from omni.physics.tensors import utils
# from omni.physx.scripts.utils import addPairFilter

# from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
# import omni.physx as physx

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# CUSTOM_GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
#     static_friction=1.2,        # coefficient of static friction
#     dynamic_friction=1.0,       # coefficient of dynamic friction
#     restitution=0.0,            # bounciness
#     # penalty stiffness (contact stiffness)
#     friction_combine_mode="multiply",
#     restitution_combine_mode="average",
#     # compliant stiffness and damping
#     compliant_contact_stiffness=1e5,
#     compliant_contact_damping=1e3,
# )




# @configclass
# class Go2SceneCfg(InteractiveSceneCfg):
#     """Configuration for a Go2 scene."""

#     # ground plane
#     ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
#                           spawn=sim_utils.GroundPlaneCfg(physics_material=CUSTOM_GROUND_MATERIAL))

#     # lights
#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )

#     # articulation
#     Go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",)


# def main():
#     """Main function."""
#     # Load kit helper
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim_cfg.dt = 0.01
#     sim_cfg.gravity = (0.0, 0.0, -9.81)
#     sim = SimulationContext(sim_cfg)
#     # Set main camera
#     sim.set_camera_view([10, 0.0, 4.0], [0.0, 0.0, 2.0])
#     # Design scene
#     scene_cfg = Go2SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)
#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     robot_positions = run_simulator(sim, scene)


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()
