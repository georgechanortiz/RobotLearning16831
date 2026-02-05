'''
    # Usage
    python scripts/my_custom_mud_environment.py --num_envs 1
'''

"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""



import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
import isaaclab.utils.math as math



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

import torch
import numpy as np
import matplotlib.pyplot as plt

CUSTOM_GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2,        # coefficient of static friction
    dynamic_friction=1.0,       # coefficient of dynamic friction
    restitution=0.0,            # bounciness
    # penalty stiffness (contact stiffness)
    friction_combine_mode="multiply",
    restitution_combine_mode="average",
    # compliant stiffness and damping
    compliant_contact_stiffness=1e5,
    compliant_contact_damping=1e3,
)


TIMESTEPS = 500


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Configuration for a Go2 scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", 
                          spawn=sim_utils.GroundPlaneCfg(physics_material=CUSTOM_GROUND_MATERIAL))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    Go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["Go2"]
    stage = get_current_stage()


    # Define simulation stepping
    sim_dt = sim.get_physics_dt() # the timestep
    count = 0

    go2_dof_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                     "FL_thigh_joint", "RL_thigh_joint", "FR_thigh_joint", "RR_thigh_joint",
                     "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
    robot_dof_idx, _ = robot.find_joints(go2_dof_names)

    # Hip, Thigh, Calf
    default_stand = torch.tensor([ 0.1, 0.1, 0.1, 0.1,
                                   0.8, 0.8, 1.0, 1.0,
                                   -1.5, -1.5, -1.5, -1.5], device=scene.device)
    

    # default_stand = torch.tensor([ 0, 0, 0, 0,
    #                                0.15, 0.15, 0.15, 0.15,
    #                                -0.3, -0.3, -0.3, -0.3], device=scene.device)
    target_pos = default_stand.repeat(scene.num_envs, 1)



    kp_hip   = 20.0
    kp_thigh = 30.0
    kp_calf_front  = 25.0
    kp_calf_rear   = 25.0

    kd_hip   = 0.8
    kd_thigh = 0.9
    kd_calf  = 0.8

    Kp_values = [kp_hip] * 4 + [kp_thigh] * 4 + [kp_calf_front] * 2 + [kp_calf_rear] *2
    Kd_values = [kd_hip] * 4 + [kd_thigh] * 4 + [kd_calf] * 4

    Kp = torch.tensor(Kp_values, device=scene.device).repeat(scene.num_envs, 1)
    Kd = torch.tensor(Kd_values, device=scene.device).repeat(scene.num_envs, 1)
    repeat = 0
    REPEAT = 500

    Tau_list = []
    while repeat < REPEAT: # simulation_app.is_running():
        repeat += 1
        joint_pos = robot.data.joint_pos[:, robot_dof_idx] # Joint Position
        joint_vel = robot.data.joint_vel[:, robot_dof_idx] # Joint Velocity

        des_joint_pos = target_pos                      # Desired Joint Position
        des_joint_vel = torch.zeros_like(target_pos)    # Desired Joint Velocity
        
        KpTau = Kp * (des_joint_pos - joint_pos)        #Proportional term
        KdTau = Kd * (des_joint_vel - joint_vel)        #Derivative term

        tau = KpTau + KdTau #+ torch.ones_like(target_pos)*9.81 #+ robot.data.projected_gravity_b[:, robot_dof_idx]
        tau = torch.clamp(tau, -30.0, 30.0)
        Tau_list.append(tau.cpu().numpy())


        #Debugging

        # print((des_joint_pos - joint_pos)[:, 0:4])    
        # print("Joint Pos", joint_pos[:,0:4])
        # print("Desired Joint Pos", des_joint_pos[:,0:4])
        # print("KP",KpTau[:,0:4])
        # print("KD",KdTau[:,0:4])
        # print("Torque", tau[:,8:12])
        # print(joint_pos)
                                    
        robot.set_joint_effort_target(tau, joint_ids=robot_dof_idx)
        robot.write_data_to_sim()


        #step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        if count % TIMESTEPS == 0:
            root_states = robot.data.default_root_state.clone()
            root_states[:, 2] += 0.01 # slightly lift the robot in spawning
            robot.write_root_state_to_sim(root_states)

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Reset buffers
            scene.reset()
            count = 0


            torch.cuda.synchronize()

            robot.update(sim_dt)
            count = 0

    Tau_array = np.array(Tau_list)

    for i in range(12):
        plt.plot(Tau_array[:, 0, i], label=f'Joint {i}')
    plt.legend()
    plt.show()
        

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
















    # while simulation_app.is_running():
        # robot.write_joint_stiffness_to_sim(torch.ones_like(target_pos) * 5.0, joint_ids=robot_dof_idx)
        # robot.write_joint_damping_to_sim(torch.ones_like(target_pos) * 0.5, joint_ids=robot_dof_idx)

        # test_effort = torch.tensor([
        #     0,0,0,
        #     0,0,0,
        #     0,10.0,0,
        #     0,10.0,0
        # ], device=scene.device).repeat(scene.num_envs, 1)
        # robot.set_joint_effort_target(test_effort, joint_ids=robot_dof_idx)



### COMMANDS TO IMPLEMENT EXTERNAL FORCES ON GO2
# robot.set_external_force_and_torque(torch.zeros_like(robot.get_external_force()), torch.zeros_like(robot.get_external_torque())) 



