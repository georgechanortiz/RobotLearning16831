"""Usage Examples:

Regular Walking Policy: 
./isaaclab.sh -p scripts/skrl/play_custom_mud.py --task=Mud-Flat-Unitree-Go2-Play-v0 --num_envs=16 --checkpoint=logs/skrl/go2_flat_ppo/2026-01-28_10-50-19_ppo_torch/checkpoints/best_agent.pt
./isaaclab.sh -p scripts/skrl/play_custom_mud.py --task=Mud-Flat-Unitree-Go2-Play-v0 --num_envs=16 --checkpoint=logs/skrl/go2_flat_ppo/2026-01-29_13-30-39_ppo_torch/checkpoints/best_agent.pt

Trained with the Forces
./isaaclab.sh -p scripts/skrl/play_custom_mud.py --task=Mud-Flat-Unitree-Go2-Play-v0 --num_envs=16 --checkpoint=logs/skrl/go2_flat_ppo/2026-02-05_10-43-12_ppo_torch/checkpoints/best_agent.pt


Walking in Rough Environment Policy:
./isaaclab.sh -p scripts/skrl/play_custom_mud.py --task=Mud-Rough-Unitree-Go2-Play-v0 --num_envs=16 --checkpoint=logs/skrl/go2_rough_ppo/2026-01-27_14-10-35_ppo_torch/checkpoints/best_agent.pt

"""

import argparse
import sys
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video",                      action="store_true", default=False, help="Record videos during training.")
# parser.add_argument("--video_length",               type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric",             action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",                   type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task",                       type=str, default=None, help="Name of the task.")
parser.add_argument("--agent",                      type=str, default=None, help=( "Name of the RL agent configuration entry point. Defaults to None, in which case the argument ""--algorithm is used to determine the default agent configuration entry point."),)
parser.add_argument("--checkpoint",                 type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed",                       type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint",  action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--ml_framework",               type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework used for training the skrl agent.")
parser.add_argument("--algorithm",                  type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm used for training the skrl agent.", )
parser.add_argument("--real-time",                  action="store_true", default=False, help="Run in real-time, if possible.")

AppLauncher.add_app_launcher_args(parser)                               # append AppLauncher cli args
args_cli, hydra_args = parser.parse_known_args()                        # parse the arguments

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args                                   # clear out sys.argv for Hydra

app_launcher = AppLauncher(args_cli)                                    # launch omniverse app
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import skrl
from packaging import version


SKRL_VERSION = "1.4.3"                                                  # check for minimum supported skrl version
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner


import isaaclab.sim as sim_utils
from isaaclab.envs import (DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent)
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# from isaaclab.utils.pretrained_checkpoint import get_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config







# import isaaclab.sensors





import mud_dynamics_2.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()




CUSTOM_GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2,                # coefficient of static friction
    dynamic_friction=1.0,               # coefficient of dynamic friction
    restitution=0.0,                    # bounciness
    friction_combine_mode="multiply",   # penalty stiffness (contact stiffness)
    restitution_combine_mode="average",
    compliant_contact_stiffness=1e5,    # compliant stiffness
    compliant_contact_damping=1e3,      # compliant damping
)





@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""


    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("Environment Config: ",env_cfg)
    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("EnvCFG Scene: ",env_cfg.scene)
    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(env_cfg.scene.contact_forces)
    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print("Received force matrix of: ", env_cfg.scene["contact_forces_LF"].data.force_matrix_w)
    # print("Received contact force of: ", env_cfg.scene["contact_forces_LF"].data.net_forces_w)
 
    print()
    #Customize ground material properties
    print(f"[INFO] Setting custom ground material properties...")
    env_cfg.scene.terrain.physics_material.static_friction = CUSTOM_GROUND_MATERIAL.static_friction
    env_cfg.scene.terrain.physics_material.dynamic_friction = CUSTOM_GROUND_MATERIAL.dynamic_friction
    env_cfg.scene.terrain.physics_material.restitution = CUSTOM_GROUND_MATERIAL.restitution  
    env_cfg.scene.terrain.physics_material.friction_combine_mode = CUSTOM_GROUND_MATERIAL.friction_combine_mode
    env_cfg.scene.terrain.physics_material.restitution_combine_mode = CUSTOM_GROUND_MATERIAL.restitution_combine_mode
    env_cfg.scene.terrain.physics_material.compliant_contact_stiffness = CUSTOM_GROUND_MATERIAL.compliant_contact_stiffness
    env_cfg.scene.terrain.physics_material.compliant_contact_damping = CUSTOM_GROUND_MATERIAL.compliant_contact_damping




    task_name = args_cli.task.split(":")[-1]                                                                   # grab task name for checkpoint path
    train_task_name = task_name.replace("-Play", "")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs    # override configurations with non-hydra CLI arguments
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.ml_framework.startswith("jax"):                                                                # configure the ML framework into the global skrl variable
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    if args_cli.seed == -1:                                                                                    # randomly sample a seed if seed = -1
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]            # set the agent and environment seed from command line
    env_cfg.seed = experiment_cfg["seed"]                                                                      # note: certain randomization occur in the environment initialization so we set the seed here
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])           # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # if args_cli.use_pretrained_checkpoint:                                                                     # get checkpoint path
    #     resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
    #     if not resume_path:
    #         print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
    #         return
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir                                                                                  # set the log directory for the environment (works for all environment types)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array") # if args_cli.video else None            # create isaac environment
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:                                      # convert to single-agent instance if required by the RL algorithm
        env = multi_agent_to_single_agent(env)
    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt



    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0




    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)


        
        # Evaluate forces on each foot of the robot


        # print(actions)





        # # ===== FORCE MONITORING =====
        # # Access the underlying environment
        # base_env = env.unwrapped

        # # Get contact sensor data
        # contact_sensor = base_env.scene.sensors["contact_forces"]

        # # Print every N steps to avoid console spam
        # if timestep % 10 == 0:  # Adjust frequency as needed
        #     print(f"\n{'='*80}")
        #     print(f"Timestep: {timestep}")
        #     print(f"{'='*80}")
            
        #     # Get all body names being tracked
        #     body_names = contact_sensor.body_names
            
        #     # Net forces on each body (shape: [num_envs, num_bodies, 3])
        #     net_forces = contact_sensor.data.net_forces_w
            
        #     # Force matrix (shape: [num_envs, num_bodies, 3])
        #     force_matrix = contact_sensor.data.force_matrix_w
            
        #     # For environment 0 (you can loop through all envs if needed)
        #     env_id = 0
            
        #     print(f"\n--- Environment {env_id} Contact Forces ---")
        #     for body_idx, body_name in enumerate(body_names):
        #         net_force = net_forces[env_id, body_idx]
        #         force_mag = torch.norm(net_force).item()
                
        #         # Only print the contact force if it exceeds a small threshold (e.g., 0.1 N)
        #         if force_mag > 0.1:
        #             print(f"{body_name}: Net Force: [{net_force[0]:.2f}, {net_force[1]:.2f}, {net_force[2]:.2f}] N  Magnitude: {force_mag:.2f} N")
                    
        #     # === FORCE DISTRIBUTION ANALYSIS ===
        #     print(f"\n--- Force Distribution Analysis (Env {env_id}) ---")
            
        #     # Separate feet and body
        #     feet_indices = []
        #     body_indices = []
            
        #     for idx, name in enumerate(body_names):
        #         if "FOOT" in name:
        #             feet_indices.append(idx)
        #         elif "base" in name.lower():
        #             body_indices.append(idx)
            
        #     # Calculate total force on feet
        #     if feet_indices:
        #         feet_forces = net_forces[env_id, feet_indices]
        #         feet_force_mags = torch.norm(feet_forces, dim=1)
        #         total_feet_force = torch.sum(feet_force_mags).item()
                
        #         print(f"\nFeet Forces:")
        #         for i, idx in enumerate(feet_indices):
        #             if feet_force_mags[i] > 0.1:  # Only print if force is above the threshold
        #                 print(f"  {body_names[idx]}: {feet_force_mags[i]:.2f} N")
        #         print(f"  Total: {total_feet_force:.2f} N")
                
        #         # Check force balance (standard deviation)
        #         if len(feet_indices) > 1:
        #             force_std = torch.std(feet_force_mags).item()
        #             force_mean = torch.mean(feet_force_mags).item()
        #             cv = (force_std / force_mean * 100) if force_mean > 0 else 0
        #             print(f"  Force Balance (CV): {cv:.1f}%")
        #             print(f"  Even Distribution: {'Yes' if cv < 30 else 'No'}")
            
        #     # Calculate total force on body
        #     if body_indices:
        #         body_forces = net_forces[env_id, body_indices]
        #         body_force_mags = torch.norm(body_forces, dim=1)
        #         total_body_force = torch.sum(body_force_mags).item()
                
        #         print(f"\nBody Forces:")
        #         for i, idx in enumerate(body_indices):
        #             if body_force_mags[i] > 0.1:  # Only print if force is above the threshold
        #                 print(f"  {body_names[idx]}: {body_force_mags[i]:.2f} N")
        #         print(f"  Total: {total_body_force:.2f} N")
            
        #     # Overall force distribution
        #     all_force_mags = torch.norm(net_forces[env_id], dim=1)
        #     total_force = torch.sum(all_force_mags).item()
            
        #     print(f"\nOverall:")
        #     print(f"  Total Contact Force: {total_force:.2f} N")
        #     if feet_indices and body_indices:
        #         feet_percentage = (total_feet_force / total_force * 100) if total_force > 0 else 0
        #         body_percentage = (total_body_force / total_force * 100) if total_force > 0 else 0
        #         print(f"  Feet: {feet_percentage:.1f}% | Body: {body_percentage:.1f}%")
            
        # timestep += 1








        # time delay for real-time evaluation   
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


    env.close()    # close the simulator

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()








