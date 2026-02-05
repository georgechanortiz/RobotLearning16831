
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

import torch

### Code will work, but will need to apply force to each agent on each leg, which will get comp expensive. Need to figure out how to optimize code so that all these forces arent going to take a million years to train.




# def suction_and_resistive_forces(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, c_range: tuple[float, float], z_threshold: float):
#     asset: ArticulationCfg = env.scene[asset_cfg.name]
#     root_pos = asset.data.body_link_pos_w[env_ids] # XYZ Position of every link wrt world origin
#     below_threshold = root_pos[:,:,2] < z_threshold # Boolean list of all positions within z threshold; "link is in mud"

#     # print(asset.data.body_link_pos_w[0,16,:]) #
#     for i in range(len(env_ids)):
#         body_ids = torch.nonzero(below_threshold[i].flatten()).squeeze().cpu().tolist() 
#         root_vel = asset.data.body_link_vel_w[env_ids]
#         root_vel = root_vel[i, body_ids, 2] # Index the root velocity using the body IDs as the index

#         suction_forces = []
#         resistive_forces = []
        
#         for k in range(len(root_vel)):
#             if root_vel[k] > 0: #suction
#                 suction_forces.append(body_ids[k])
#             else:               #resistive
#                 resistive_forces.append(body_ids[k])

#         if len(suction_forces) > 0:
#             # Create tensor
#             forces = torch.zeros((len(suction_forces), 1, 3), device=asset.device)
#             forces[:, 0, 2] = torch.FloatTensor(len(suction_forces)).uniform_(*c_range).to(asset.device)  #assign random value within range of coefficient
#             for j in range(len(suction_forces)):                                                          # Loops through each leg ID, applies the quadratic equation to it and multiplies that with the forces 
#                 curr_z_for_affected_ids = root_pos[0,suction_forces[j], 2]                                # Takes z and makes the z a function of the 
#                 forces[j, 0, 2] = -forces[j, 0, 2] * ((z_threshold - curr_z_for_affected_ids)/z_threshold)**2      # Quadratic conversion 
#             # Apply force
#             torques = torch.zeros_like(forces)        
#             asset.set_external_force_and_torque(forces=forces, torques=torques, env_ids=env_ids[i:i+1], body_ids=suction_forces) 
#             asset.write_data_to_sim()

#         if len(resistive_forces) > 0:
#             # Create tensor
#             forces = torch.zeros((len(resistive_forces), 1, 3), device=asset.device)
#             forces[:, 0, 2] = torch.FloatTensor(len(resistive_forces)).uniform_(*c_range).to(asset.device)  
#             for j in range(len(resistive_forces)): 
#                 curr_z_for_affected_ids = root_pos[0,resistive_forces[j], 2]     
#                 forces[j, 0, 2] = forces[j, 0, 2] * ((z_threshold - curr_z_for_affected_ids)/z_threshold)**2     
#             torques = torch.zeros_like(forces)        
#             asset.set_external_force_and_torque(forces=forces, torques=torques, env_ids=env_ids[i:i+1], body_ids=resistive_forces) 
#             asset.write_data_to_sim()




    # Can approximate shear as a point load on the foot, as Go2 will only be able to read torque state at joints for shear. Can derive some equation to relate the shear to torque, get some approximation then domain rand it 




# def shear_forces(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, c1: tuple[float, float], c2: tuple[float, float], z_threshold: float, area: float):
#     asset: ArticulationCfg = env.scene[asset_cfg.name]
#     root_pos = asset.data.body_link_pos_w[env_ids]    # root_pos: [num_envs, num_bodies, 3]
#     root_vel = asset.data.body_link_vel_w[env_ids]    # root_vel: [num_envs, num_bodies, 3]
    
#     # Extract positions and velocities
#     z_pos = root_pos[..., 2]   
#     xy_pos = root_pos[..., 0:2]   
#     xy_vel = root_vel[..., 0:2]    

#     # Calculate norm of each vector
#     pos_magnitude = torch.linalg.norm(xy_pos, dim=2, keepdim=True)
#     vel_magnitude = torch.linalg.norm(xy_vel, dim=2, keepdim=True)

#     pos_magnitude = pos_magnitude[:,-4:]
#     vel_magnitude = vel_magnitude[:,-4:]

#     angle = torch.atan2(xy_vel[..., 1], xy_vel[..., 0])# Angle of velocity vector in XY plane
#     angle = angle[:,-4:]
#     angle = angle + torch.pi


#     in_mud = z_pos < z_threshold    # Bodies inside mud (Bool)
#     shear_rate = vel_magnitude / pos_magnitude
#     m = (c1[1] - c1[0]) * torch.rand(1) + c1[0]
#     b = (c2[1] - c2[0]) * torch.rand(1) + c2[0]
#     m = m.to(asset.device)
#     b = b.to(asset.device)

#     shear_stress = .2 * torch.log(shear_rate + 1e-3) + m * shear_rate + b # Shear stress on sides of the leg 

#     forces = torch.zeros((len(env_ids), 19, 3), device='cuda:0')

#     forces[..., -4:, 0] = shear_stress.squeeze() * torch.cos(angle).squeeze() * area * in_mud[:, -4:].squeeze()   # X component
#     forces[..., -4:, 1] = shear_stress.squeeze() * torch.sin(angle).squeeze() * area * in_mud[:, -4:].squeeze()   # Y component



#     asset.instantaneous_wrench_composer.set_forces_and_torques(forces=forces, torques=torch.zeros_like(forces), env_ids=env_ids, body_ids=None) # Set and write forces
#     asset.write_data_to_sim()


