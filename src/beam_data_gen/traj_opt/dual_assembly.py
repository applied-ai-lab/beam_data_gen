from typing import List

import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import mujoco
from filterpy.monte_carlo import systematic_resample
from tqdm import trange
import mujoco
from scipy.spatial.transform import Rotation as R

from beam_data_gen.traj_opt.traj_opt_base import (TrajOptParams, TrajOptBase)
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim


class ParticleTrajectories:
    def __init__(self):
        self.particles: torch.Tensor 
        self.gripper_particles: torch.Tensor
        self.indices: torch.Tensor
        self.no_live_particles = torch.Tensor
        self.loss = torch.Tensor
        
    def sample_indices(self) -> List[int]:
        traj_len = self.particles.shape[1]
        
        indices = [0] * traj_len
        particle_idx = np.random.choice(self.indices[-1, :])
        
        for k in range(traj_len - 1, -1, -1):
            particle_idx = self.indices[k, particle_idx].item()
            indices[k] = particle_idx
        
        return indices            
    
    def sample_trajectories(self, indices) -> torch.Tensor:
        _, time_steps, no_features = self.particles.shape
        particles = torch.zeros([time_steps, no_features], dtype=self.particles.dtype).to(self.particles.device)
        gripper = torch.zeros([time_steps, self.gripper_particles.shape[2]], dtype=self.particles.dtype).to(self.particles.device)
        for k, idx in enumerate(indices):
            particles[k, :] = self.particles[idx, k, :]
            gripper[k, :] = self.gripper_particles[idx, k, :]
        return particles, gripper
    
class StateParams:
    def __init__(self, state_dim:int, no_beams:int, no_hands:int, no_pins:int, device:torch.device, tol: float):
        self.device = device
        self.no_hands = no_hands
        self.no_beams = no_beams
        self.no_pins = no_pins
        self.state_dim = state_dim
        self.tol = tol


class DualArmStates:
    def __init__(self, params: StateParams):
        self.params = params
        
        # Hand poses
        self.left_pose = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        self.right_pose = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Hand start poses
        self.left_start = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        self.right_start = torch.zeros(self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Beam poses
        self._beam_poses = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Beam goals
        self._beam_goal = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)
        
        # Pregrasp locations
        self._pregrasp = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).requires_grad_(True).to(self.params.device)
        
        # Pregrasp goals
        self._pregrasp_goal = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)

        self.offset = torch.tensor([0, 0, 0.08, 0, 0], dtype=torch.float32)
        self.grasp_offset = self.offset.repeat(self.params.no_beams).view(-1, self.params.state_dim).to(self.params.device)
        
    def initialise(self):
        with torch.no_grad():
            self.pregrasp = self.beam_poses.view(self.pregrasp.shape).clone().detach() + self.grasp_offset.view(self.pregrasp.shape)
        self.pregrasp.requires_grad_(True)
        return        
        
    def advance(self):
        # Update the forward dynamics of the states
        with torch.no_grad():
            self.pregrasp.view(self.params.no_beams, -1)[:, 0:2] = self.beam_poses.view(self.params.no_beams, -1)[:, 0:2].clone().detach()
            self.pregrasp.view(self.params.no_beams, -1)[:, 3:] = self.beam_poses.view(self.params.no_beams, -1)[:, 3:].clone().detach()
        return 
    
    def requires_grad(self):
        self.left_pose.requires_grad_(True)
        self.right_pose.requires_grad_(True)
        self.beam_poses.requires_grad_(True)
        self.pregrasp.requires_grad_(True)
        return
        
    
    def detach(self):
        self.left_pose.detach()
        self.right_pose.detach()
        self.beam_poses.detach()
        self.pregrasp.detach()
        return
    
    # Helper funcs
    def pose_quat_to_state(self, pose_quat_xyzw: np.array):
        rows, _ = pose_quat_xyzw.shape
        state = torch.zeros((rows, self.params.state_dim), dtype=torch.float32).to(self.params.device)
        
        for k in range(rows):
            state[k, 0:3] = pose_quat_xyzw[k, 0:3]
            
            rot = R.from_quat(pose_quat_xyzw[k, 3:])
            
            state[k, 3] = torch.sin(rot.as_euler(seq="xyz")[2])
            state[k, 4] = torch.cos(rot.as_euler(seq="xyz")[2])
            
        return state
    
    # Beam Poses setters and getters
    @property
    def beam_poses(self):
        return self._beam_poses
    
    @beam_poses.setter
    def beam_poses(self, value):
        self._beam_poses = value.view(-1, self.params.state_dim)
    
    # Beam Goals setters and getters
    @property
    def beam_goal(self):
        return self._beam_goal
    
    @beam_goal.setter
    def beam_goal(self, value):
        self._beam_goal = value.view(-1, self.params.state_dim)
    
    # Pregrasp Poses setters and getters
    @property
    def pregrasp(self):
        return self._pregrasp
    
    @pregrasp.setter
    def pregrasp(self, value):
        self._pregrasp = value.view(-1, self.params.state_dim)        
    
    # Pregrasp Goal setters and getters
    @property
    def pregrasp_goal(self):
        self._pregrasp_goal = self.beam_poses.view(self._pregrasp_goal.shape) + self.grasp_offset.view(self._pregrasp_goal.shape)
        return self._pregrasp_goal
    
    @pregrasp_goal.setter
    def pregrasp_goal(self, value):
        self._pregrasp_goal = value.view(-1, self.params.state_dim)
        return
    

class DualArmGradients:
    def __init__(self, params: StateParams):
        self.params = params
        
        # Hand poses
        self.left_pose = torch.zeros(self.params.state_dim).to(self.params.device)
        self.right_pose = torch.zeros(self.params.state_dim).to(self.params.device)
        
        # Beam poses
        self.beam_poses = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)
        
        # Pregrasp locations
        self.pregrasp = torch.zeros(self.params.no_beams * self.params.state_dim).view(-1, self.params.state_dim).to(self.params.device)
        
    def zero(self):
        self.left_pose *= 0.
        self.right_pose *= 0.
        self.beam_poses *= 0.
        self.pregrasp *= 0.
        return

class LossesContacts:
    def __init__(self, params: StateParams):
        self.params = params
        
        self._mse_sum = nn.MSELoss(reduction="sum")
        self._mse_none = nn.MSELoss(reduction='none')
        
    def calc_losses(self, states: DualArmStates):
        pass
    
    def calc_prob(self):
        pass
        
        
class HandLossesContacts(LossesContacts):
    def __init__(self, params):
        super().__init__(params)
        
        self._pregrasp_con = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._pregrasp_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
        self._beam_con = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        self._beam_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
        self._start_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
        # Beam target losses
        self._beam_conver_loss = torch.zeros(self.params.no_beams).to(self.params.device)
        self._beam_conver_p = torch.zeros(self.params.no_beams).to(self.params.device)
        # Pregrasp target losses
        self._pregrasp_conver_loss = torch.zeros(self.params.no_beams).to(self.params.device)
        self._pregrasp_conver_p = torch.zeros(self.params.no_beams).to(self.params.device)
        
    def calc_losses(self, states: DualArmStates):
        self._pregrasp_loss[0:self.params.no_beams] = self._mse_none(states.pregrasp.view(self.params.no_beams, -1), 
                                            states.left_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._pregrasp_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.pregrasp.view(self.params.no_beams, -1), 
                                            states.right_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        
        self._beam_loss[0:self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1), 
                                            states.left_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._beam_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1), 
                                            states.right_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        
        self._start_loss[0:self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1), 
                                            states.left_start.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._start_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1), 
                                            states.right_start.repeat(self.params.no_beams, 1)).sum(dim=1)
        
        # Check which means are at target locations
        self._beam_conver_loss = self._mse_none(states.beam_poses, states.beam_goal).sum(1)
        self._pregrasp_conver_loss = self._mse_none(states.pregrasp, states.pregrasp_goal).sum(1)
        return
    
    def calc_prob(self):
        self._pregrasp_con = (self._pregrasp_loss < self.params.tol).type(torch.float32)
        # Which beams are in contact with the 
        self._beam_con = (self._beam_loss < self.params.tol).type(torch.float32)  
        # Check which beams are converged 
        self._beam_conver_p = (self._beam_conver_loss < self.params.tol).type(torch.float32)
        self._pregrasp_conver_p = (self._pregrasp_conver_loss < self.params.tol).type(torch.float32)
        
        # Check if the beams have converged
        for k in range(self.params.no_beams):
            # If converged hand is not in contact
            if self._beam_conver_p[k] > 0.5:
                print( f" Release hand for beam {k} " )
                self._beam_con[k] = 0.0
                self._beam_con[self.params.no_beams + k] = 0.0
        return
    
    
class BeamLosses(LossesContacts):
    def __init__(self, params):
        super().__init__(params)        
        self._beam_losses = torch.zeros(self.params.no_beams * self.params.state_dim).to(self.params.device)
        
    def calc_losses(self, states):
        self._beam_losses = self._mse_sum(states.beam_poses, states.beam_goal)
        return
        
class PregraspLosses(LossesContacts):
    def __init__(self, params):
        super().__init__(params)
        
        self.pregrasp_loss = torch.zeros(self.params.no_hands * self.params.no_beams).to(self.params.device)
        
    def calc_losses(self, states):
        self.pregrasp_loss = self._mse_sum(states.pregrasp, states.pregrasp_goal)
        return
    

class DualAssembly(TrajOptBase):
    def __init__(self, 
                params: TrajOptParams, 
                state_params: StateParams, 
                sim: SquareRobotSim,
                left_start: torch.Tensor,
                right_start: torch.Tensor, 
                model: mujoco.MjModel,
                data: mujoco.MjData):
        
        super().__init__(params)
        
        self._state_params = state_params
        self.state_dim = self._state_params.state_dim
        
        # States
        self._states = DualArmStates(self._state_params)
        # Gradients 
        self._gradients = DualArmGradients(self._state_params)
        
        # Losses
        self._hand_losses = HandLossesContacts(self._state_params)
        self._beam_losses = BeamLosses(self._state_params)
        self._pregrasp_losses = PregraspLosses(self._state_params)
        
        self.sim = sim
        
        self._left_index = -1
        self._right_index = -1
        
        if sim is not None:
            self.node_names = list(self.sim._geom_to_name.values())
        
        self.left_loss = None
        self.right_loss = None
        
        self.left_start = left_start
        self.right_start = right_start
        
        # Noise weights 
        self.w = 0.0 * torch.tensor([0.01, 0.01, 0.01, 0.2, 0.2], dtype=torch.float32).to(self.params.device)
        
        # Convergence dict
        self._convergence = {}
        
        # Mujoco pointers
        self._mu_model = model
        self._mu_data = data
        
        # Containers for solutions
        self._particle_trajectories = None        
        
    def initialise(self, states: DualArmStates):
        self._states = states
        # Create container
        self._particle_trajectories = ParticleTrajectories()
        # Allocate container
        self._particle_trajectories.particles = torch.zeros([self.params.no_particles, self.params.no_steps, self._x.shape[0]], dtype=torch.float32).to(self._x.device)
        self._particle_trajectories.gripper_particles = torch.zeros([self.params.no_particles, self.params.no_steps, self._state_params.no_hands], dtype=torch.float32).to(self._x.device)
        self._particle_trajectories.indices = torch.zeros([self.params.no_steps, self.params.no_particles], dtype=torch.int32)
        self._particle_trajectories.no_live_particles = torch.zeros([self.params.no_steps], dtype=torch.int32)
        self._particle_trajectories.loss = torch.zeros([self.params.no_particles, self.params.no_steps], dtype=torch.float32)
        # Weights
        self._weights = np.zeros([self.params.no_particles])
        return      
    
    def reset(self):
        self._particle_trajectories.particles.zero_()
        self._particle_trajectories.indices.zero_()
        self._particle_trajectories.no_live_particles.zero_()
        self._particle_trajectories.loss.zero_()
        self._weights *= 0.0
        return
        
    
    def optimise(self) -> ParticleTrajectories:
        
        # Create or reset the containers
        if self._particle_trajectories is None:
            self.initialise(self.states)
        else:
            self.reset()
            
        # Set the initial particles to the states
        for i in range(self._params.no_particles):
            self._particle_trajectories.particles[i, 0, :] = self._x
        
        for k in range(1, self._params.no_steps, 1):
            for n in range(self._params.no_particles):
                
                self._states.requires_grad()
            
                gradients = self.gradients()
                
                # Update the states
                self._states.left_pose = self._states.left_pose - self.params.step_size * gradients.left_pose
                self._states.right_pose = self._states.right_pose - self.params.step_size * gradients.right_pose
                self._states.beam_poses = self._states.beam_poses - self.params.step_size * gradients.beam_poses
                self._states.pregrasp = self._states.pregrasp - self.params.step_size * gradients.pregrasp         
                
                self._states.detach()       
                
                self._particle_trajectories.loss[n, k] = self._beam_losses._beam_losses.sum()
                
                # Apply noise to gradients
                self._x = torch.cat([self._states.left_pose.reshape(-1), 
                                    self._states.right_pose.reshape(-1), 
                                    self._states.beam_poses.reshape(-1),
                                    self._states.pregrasp.reshape(-1)], dim=0)
                
                self._x[0:(self._state_params.no_beams + self._state_params.no_hands) * self.state_dim] = self.normalise_pose(self._x[0:(self._state_params.no_beams + self._state_params.no_hands) * self.state_dim])
                self._particle_trajectories.particles[n, k, :] = self._x
                # Gripper state
                self._particle_trajectories.gripper_particles[n, k, 0] = (self._hand_losses._beam_con[0:self._state_params.no_beams] > 0.5).any().type(torch.float32)
                self._particle_trajectories.gripper_particles[n, k, 1] = (self._hand_losses._beam_con[self._state_params.no_beams: 2 * self._state_params.no_beams] > 0.5).any().type(torch.float32)

                
                # Check collisions
                self.sim.decode_x(self._mu_data, self._x[0:(self._state_params.no_hands + self._state_params.no_beams) * self.state_dim].unsqueeze(0))
                mujoco.mj_step(self._mu_model, self._mu_data)
                
                # Check for collisions with moving beams
                if self.sim.check_collisions(self._mu_data, self.node_names[self._left_index]) or self.sim.check_collisions(self._mu_data, self.node_names[self._right_index]):
                    self._weights[n] = 1.0e-5
                else:
                    self._weights[n] = 1.0
                    self._particle_trajectories.no_live_particles[k] += 1
            
            # Resample particles
            self._weights /= self._weights.sum()
            
            indices = systematic_resample(self._weights)
            self._particle_trajectories.particles[:, k, :] = self._particle_trajectories.particles[indices, k, :]
            self._particle_trajectories.gripper_particles[:, k, :] = self._particle_trajectories.gripper_particles[indices, k, :]
            self._particle_trajectories.indices[k] = torch.tensor(indices)                
        
        return self._particle_trajectories
    
    def gradients(self):
        # Calc losses
        self._hand_losses.calc_losses(self._states)
        self._hand_losses.calc_prob()
        
        left_contacts = self._hand_losses._beam_con[0:self._state_params.no_beams]
        right_contacts = self._hand_losses._beam_con[self._state_params.no_beams:2*self._state_params.no_beams]
        
        left_pregrasp_c = self._hand_losses._pregrasp_con[0:self._state_params.no_beams]
        right_pregrasp_c = self._hand_losses._pregrasp_con[self._state_params.no_beams:2 * self._state_params.no_beams]
        
        self._beam_losses.calc_losses(self._states)
        self._pregrasp_losses.calc_losses(self._states)
        
        # Calculate beam gradients
        self._gradients.beam_poses = grad(outputs=self._beam_losses._beam_losses, inputs=self.states.beam_poses, retain_graph=True)[0]
        
        self.left_loss = self._hand_losses._beam_loss[0:self._state_params.no_beams] * (left_pregrasp_c)  + \
                            self._hand_losses._pregrasp_loss[0:self._state_params.no_beams] * (1 - left_pregrasp_c)
                            
        self.right_loss = self._hand_losses._beam_loss[self._state_params.no_beams:2*self._state_params.no_beams] * (right_pregrasp_c)  + \
                            self._hand_losses._pregrasp_loss[self._state_params.no_beams: 2*self._state_params.no_beams] * (1 - right_pregrasp_c)

        # If converged return zeros for gradients
        if self.check_convergence(self._hand_losses._beam_conver_p,
                                self._hand_losses._pregrasp_conver_p, 
                                self.left_loss,
                                self.right_loss):
            self._gradients.zero()
            print(" Converged ")
            return self._gradients
        
        self._gradients.beam_poses = (self._gradients.beam_poses * self._hand_losses._beam_con[0:self._state_params.no_beams].reshape(self._gradients.beam_poses.shape[0], 1) + \
                                        self._gradients.beam_poses * self._hand_losses._beam_con[self._state_params.no_beams:2*self._state_params.no_beams].reshape(self._gradients.beam_poses.shape[0], 1))
        # Apply noises
        noise = torch.randn_like(self._gradients.beam_poses)
        self._gradients.beam_poses += min(torch.norm(self._gradients.beam_poses, p=2.0), 1.0) * noise * self.w
        self._gradients.beam_poses = (self._gradients.beam_poses * left_contacts.reshape(self._gradients.beam_poses.shape[0], 1) + self._gradients.beam_poses * right_contacts.reshape(self._gradients.beam_poses.shape[0], 1))
        
        self._gradients.beam_poses[:, 3:5] *= 1.5
        
        # Hand gradients
        self._gradients.left_pose = grad(self.left_loss[self._left_index], inputs=self.states.left_pose, retain_graph=True)[0] * (1. - left_contacts[self._left_index]) + \
                                    self._gradients.beam_poses[self._left_index, :]
        self._gradients.right_pose = grad(self.right_loss[self._right_index], inputs=self.states.right_pose, retain_graph=True)[0] * (1. - right_contacts[self._right_index]) + \
                                    self._gradients.beam_poses[self._right_index, :]
        
        # Calculate the gradients for the pregrasp pose
        self._gradients.pregrasp = grad(outputs=self._pregrasp_losses.pregrasp_loss, inputs=self._states.pregrasp, retain_graph=True)[0]
        
        # If beam has converged
        if self._hand_losses._beam_conver_p[self._left_index] > 0.5:
            self._gradients.left_pose = self._gradients.pregrasp[self._left_index, :]
        else:
            self._gradients.pregrasp[self._left_index] = self._gradients.left_pose * left_pregrasp_c[self._left_index]
        
        if self._hand_losses._beam_conver_p[self._right_index] > 0.5:            
            self._gradients.right_pose = self._gradients.pregrasp[self._right_index, :]
        else:
            self._gradients.pregrasp[self._right_index] = self._gradients.right_pose * right_pregrasp_c[self._right_index] 
        
        return self._gradients
    
    def check_convergence(self, beam_conv_p, pregrasp_conv_p, left_loss, right_loss):
                
        self.active_left_loss = self._hand_losses._start_loss[0: self._state_params.no_beams].clone()
        self.active_right_loss = self._hand_losses._start_loss[self._state_params.no_beams: 2 * self._state_params.no_beams].clone()
        
        # Pin penalties
        pin_indices = list(2 * k + 1 for k in range(self._state_params.no_pins))
        
        self.active_left_loss[pin_indices] *= 10.0
        self.active_right_loss[pin_indices] *= 10.0
        
        for k in range(self._state_params.no_beams):
            if beam_conv_p[k] > 0.5 and pregrasp_conv_p[k] > 0.5:
                
                self._convergence[k] = True
                # Remove the index
                self.active_left_loss[k] *= 1.e6
                self.active_right_loss[k] *= 1.e6
                
        if len(self._convergence.keys()) == self._state_params.no_beams:
            return True
        
        self._left_index = torch.argmin(self.active_left_loss, 0)
        self._right_index = torch.argmin(self.active_right_loss, 0)

        # Figure out what to do if a beam is equi-distant
        while self._right_index == self._left_index:
                        
            self.active_right_loss[self._right_index] *= 1.e6
            self._right_index = torch.argmin(self.active_right_loss, 0)
            
        return False
    
    def normalise_pose(self, pose_torch: torch.tensor):
        no_items = pose_torch.shape[0] // self.state_dim    
        for k in range(no_items):
            pose_torch[self.state_dim * k + 2] = max(pose_torch[self.state_dim * k + 2], 0.021)
            pose_torch[self.state_dim * k + 3: self.state_dim * k + 5] = \
                torch.nn.functional.normalize(pose_torch[self.state_dim * k + 3: self.state_dim * k + 5], dim=0)
        return pose_torch
    
    @property
    def particle_trajectories(self):
        if self._x is not None:
            self.initialise()
        else:
            print(" Please set_x to first to allocate data. ")
    
    @property
    def states(self):
        return self._states
    
    @states.setter
    def states(self, state_values: DualArmStates):
        self._states = state_values
        self._states.advance()
        # Update x
        self._x = torch.cat([state_values.left_pose.reshape(-1), 
                            state_values.right_pose.reshape(-1), 
                            state_values.beam_poses.reshape(-1), 
                            state_values.pregrasp.reshape(-1)], dim=0)
        return