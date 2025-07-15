from typing import List

import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import mujoco
from filterpy.monte_carlo import systematic_resample
from tqdm import trange
import mujoco

from beam_data_gen.traj_opt.traj_opt_base import (TrajOptParams, TrajOptBase)
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim


class ParticleTrajectories:
    def __init__(self):
        self.particles: torch.Tensor 
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
        for k, idx in enumerate(indices):
            particles[k, :] = self.particles[idx, k, :]
        return particles
    
class StateParams:
    def __init__(self, state_dim:int, no_beams:int, no_hands:int, device:torch.device, tol: float):
        self.device = device
        self.no_hands = no_hands
        self.no_beams = no_beams
        self.state_dim = state_dim
        self.tol = tol


class DualArmStates:
    def __init__(self, params: StateParams):
        self.params = params
        
        # Hand poses
        self.left_pose = torch.zeros(self.params.state_dim).to(self.params.device)
        self.right_pose = torch.zeros(self.params.state_dim).to(self.params.device)
        
        # Beam poses
        self.beam_poses = torch.zeros(self.params.no_beams * self.params.state_dim).to(self.params.device)
        
        # Beam goals
        self.beam_goal = torch.zeros(self.params.no_beams * self.params.state_dim).to(self.params.device)
        
        # Pregrasp locations
        self.pregrasp = torch.zeros(self.params.no_beams * self.params.state_dim).to(self.params.device)
        
        # Pregrasp goals
        self.pregrasp_goal = torch.zeros(self.params.no_beams * self.params.state_dim).to(self.params.device)


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
        
    def calc_losses(self, states: DualArmStates):
        self._pregrasp_loss[0:self.params.no_beams] = self._mse_none(states.pregrasp.view(self.params.no_beams, -1), 
                                            states.left_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._pregrasp_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.pregrasp.view(self.params.no_beams, -1), 
                                            states.right_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        
        self._beam_loss[0:self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1), 
                                            states.left_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        self._beam_loss[self.params.no_beams: 2 * self.params.no_beams] = self._mse_none(states.beam_poses.view(self.params.no_beams, -1), 
                                            states.right_pose.repeat(self.params.no_beams, 1)).sum(dim=1)
        
        return
    
    def calc_prob(self):
        self._pregrasp_con = (self._pregrasp_loss < self.params.tol).type(torch.float32)
        self._beam_con = (self._beam_loss < self.params.tol).type(torch.float32)        
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
                state_dim: int, 
                sim: SquareRobotSim,
                left_start: torch.Tensor,
                right_start: torch.Tensor, 
                model: mujoco.MjModel,
                data: mujoco.MjData):
        
        super().__init__(params)
        self.state_dim = state_dim
        
        self.sim = sim
        
        self._left_index = -1
        self._right_index = -1
        
        self.node_names = list(self.sim._geom_to_name.values())
        
        self.left_loss = None
        self.right_loss = None
        
        self.left_start = left_start
        self.right_start = right_start
        
        # Noise weights 
        self.w = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2], dtype=torch.float32).to(self.params.device)
        
        # Convergence dict
        self._convergence = {}
        
        # Quantities
        self._no_hands = 2
        # Losses
        self._beam_loss = nn.MSELoss(reduction="sum")
        self._hand_loss = nn.MSELoss(reduction='none')  
        
        # Mujoco pointers
        self._mu_model = model
        self._mu_data = data
        
        # Containers for solutions
        self._particle_trajectories = None        
        
    def initialise(self, left_hand, right_hand, beams):
        self.c = beams.clone()
        # Set the beam and hand states
        self.set_x(left_hand, right_hand, beams)
        # Create container
        self._particle_trajectories = ParticleTrajectories()
        # Allocate container
        self._particle_trajectories.particles = torch.zeros([self.params.no_particles, self.params.no_steps, self._x.shape[0]], dtype=self._x.dtype).to(self._x.device)
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
        self._weights.zero_()
        return
        
    
    def optimise(self) -> ParticleTrajectories:
        
        # Create or reset the containers
        if self._particle_trajectories is None:
            self.initialise()
        else:
            self.reset()
        
        for k in trange(self._params.no_steps):
            for n in range(self._params.no_particles):
            
                x_grads, beam_losses = self._gradients(1.e-3)
                
                self._particle_trajectories.loss[n, k] = beam_losses.sum()
                
                # Apply noise to gradients
                self._x = self._x - self.params.step_size * x_grads
                
                self._x = self.normalise_pose(self._x)
                self._particle_trajectories.particles[n, k, :] = self._x
                
                # Check collisions
                self.sim.decode_x(self._mu_data, self._x.unsqueeze(0))
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
            self._particle_trajectories.indices[k] = torch.tensor(indices)
        
        return self._particle_trajectories
    
    def _gradients(self, tol: float):
        
        # Pin penalty
        pin_indices = list(2 * k + 1 for k in range(4))
        
        left_hand = self._x[0:self.state_dim]
        right_hand = self._x[self.state_dim: 2 * self.state_dim]
        beam_poses = self._x[self._no_hands * self.state_dim: ].view(-1, self.state_dim)
        # Update the pregrasp poses
        c_target = beam_poses.clone() 
        no_beams = beam_poses.shape[0] // self.state_dim
        z_offset = 0.08
        for k in range(no_beams):
            c_target[k * self.state_dim + 2] += z_offset
        
        # Calculate losses
        beam_losses = self._beam_loss(beam_poses, self.goal.view(-1, self.state_dim))
        # Hand losses
        left_beam_loss = self._hand_loss(beam_poses, left_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        right_beam_loss = self._hand_loss(beam_poses, right_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        
        left_pregrasp_loss = self._hand_loss(self.c, left_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        right_pregrasp_loss = self._hand_loss(self.c, right_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        
        # Calculate gradients
        beam_gradients = grad(outputs=beam_losses, inputs=beam_poses, retain_graph=True)[0]
        
        # Check if any of the goals have converged
        self.check_convergence(beam_gradients)        
        
        # Use loss to calculate beam contacts
        left_contacts = (left_beam_loss < tol).type(torch.float32)
        right_contacts = (right_beam_loss < tol).type(torch.float32)
        
        # Pregrasp contacts
        left_pregrasp_c = (left_pregrasp_loss < tol).type(torch.float32)
        right_pregrasp_c = (right_pregrasp_loss < tol).type(torch.float32)
        
        self.left_loss = left_beam_loss * (left_contacts) + left_pregrasp_loss * (1 - left_contacts)
        self.right_loss = right_beam_loss * (right_contacts) + right_pregrasp_loss * (1 - right_contacts)
        
        self.left_loss[pin_indices] *= 10.0
        self.right_loss[pin_indices] *= 10.0
        
        # Find pregrasp losses
        c_loss = self._hand_loss(self.c, c_target).sum(dim=1) * (1 - left_pregrasp_c) + self.left_loss * left_pregrasp_c \
                        + self._hand_loss(self.c, c_target).sum(dim=1) * (1 - right_pregrasp_c) + self.right_loss * right_pregrasp_c
        
        c_grad = grad(outputs=c_loss, inputs=self.c, retain_graph=True)[0]
        
        left_start_loss = self._hand_loss(beam_poses, self.left_start.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        right_start_loss = self._hand_loss(beam_poses, self.right_start.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        
        # Find smallest gradients
        self._right_index = torch.argmin(self.right_loss + right_start_loss, 0)
        self._left_index = torch.argmin(self.left_loss + left_start_loss, 0)
        
        # Figure out what to do if a beam is equi-distant
        if self._right_index == self._left_index:
            self.right_loss[self._right_index] = 1.0e3
            self._right_index = torch.argmin(self.right_loss, 0)
        
        beam_gradients = (beam_gradients * left_contacts.reshape(beam_gradients.shape[0], 1) + beam_gradients * right_contacts.reshape(beam_gradients.shape[0], 1))
        # Apply noises
        noise = torch.randn_like(beam_gradients)
        beam_gradients += min(torch.norm(beam_gradients, p=2.0), 1.0) * noise * self.w
        beam_gradients = (beam_gradients * left_contacts.reshape(beam_gradients.shape[0], 1) + beam_gradients * right_contacts.reshape(beam_gradients.shape[0], 1))
        
        beam_gradients[:, 3:5] *= 2.0
        
        # Hand gradients
        left_gradients = grad(self.left_loss[self._left_index], inputs=left_hand, retain_graph=True)[0] * (1. - left_contacts[self._left_index]) + beam_gradients[self._left_index, :]
        right_gradients = grad(self.right_loss[self._right_index], inputs=right_hand, retain_graph=True)[0] * (1. - right_contacts[self._right_index]) + beam_gradients[self._right_index, :]
        
        return torch.cat([left_gradients, right_gradients, beam_gradients.view(-1), c_grad.view(-1)], dim=0), beam_losses
    
    def check_convergence(self, gradient):
        
        grad_norm = torch.norm(gradient, p=2.0, dim=1)
        # Item with largest gradient
        index = torch.argmin(grad_norm)
        min_val = grad_norm[index]
        
        while min_val < self.params.epsilon:
            
            self._convergence[index.item()] = True
            
            if len(self._convergence.keys()) >= gradient.shape[0]:
                gradient *= 0.0
                break
            
            gradient[index, :] *= 1.0e6      
            
            self.left_loss[index] *= 1.0e6
            self.right_loss[index] *= 1.0e6
            
            grad_norm = torch.norm(gradient, p=2.0, dim=1)
            index = torch.argmin(grad_norm)
            min_val = grad_norm[index]           
            
        return
    
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
        
    
    def set_x(self, left_hand, right_hand, beams):
        self.c.view(-1, self.state_dim)[:, 0:2] = beams.view(-1, self.state_dim)[:, 0:2]
        self.c.view(-1, self.state_dim)[:, 3:] = beams.view(-1, self.state_dim)[:, 3:]
        
        self._x = torch.cat([left_hand, right_hand, beams, self.c], dim=0)
        return 