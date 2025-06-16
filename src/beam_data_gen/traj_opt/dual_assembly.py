from typing import List

import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import mujoco
from filterpy.monte_carlo import systematic_resample
from tqdm import trange

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
        

class DualAssembly(TrajOptBase):
    def __init__(self, 
                params: TrajOptParams, 
                state_dim: int, 
                sim: SquareRobotSim,
                left_start: torch.Tensor,
                right_start: torch.Tensor):
        
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
    
    def optimise(self, model, data) -> ParticleTrajectories:
        
        part_traj = ParticleTrajectories()
        
        part_traj.particles = torch.zeros([self.params.no_particles, self.params.no_steps, self._x.shape[0]], dtype=self._x.dtype).to(self._x.device)
        part_traj.indices = torch.zeros([self.params.no_steps, self.params.no_particles], dtype=torch.int32)
        part_traj.no_live_particles = torch.zeros([self.params.no_steps], dtype=torch.int32)
        part_traj.loss = torch.zeros([self.params.no_particles, self.params.no_steps], dtype=torch.float32)
        
        weights = np.zeros([self.params.no_particles])
        
        for k in trange(self._params.no_steps):
            for n in range(self._params.no_particles):
            
                x_grads, beam_losses = self._gradients(1.e-3)
                
                part_traj.loss[n, k] = beam_losses.sum()
                
                # Apply noise to gradients
                self._x = self._x - self.params.step_size * x_grads
                
                self._x = self.normalise_pose(self._x)
                part_traj.particles[n, k, :] = self._x
                
                # Check collisions
                self.sim.decode_x(data, self._x.unsqueeze(0))
                mujoco.mj_step(model, data)
                
                # Check for collisions with moving beams
                if self.sim.check_collisions(data, self.node_names[self._left_index]) or self.sim.check_collisions(data, self.node_names[self._right_index]):
                    weights[n] = 1.0e-5
                else:
                    weights[n] = 1.0
                    part_traj.no_live_particles[k] += 1
            
            # Resample particles
            weights /= weights.sum()
            
            indices = systematic_resample(weights)
            part_traj.particles[:, k, :] = part_traj.particles[indices, k, :]
            part_traj.indices[k] = torch.tensor(indices)
        
        return part_traj
    
    
    def _gradients(self, tol: float):
        
        # Pin penalty
        pin_indices = list(2 * k + 1 for k in range(4))
        
        left_hand = self._x[0:self.state_dim]
        right_hand = self._x[self.state_dim: 2 * self.state_dim]
        beam_poses = self._x[self._no_hands * self.state_dim: ].view(-1, self.state_dim)
        # Calculate losses
        beam_losses = self._beam_loss(beam_poses, self.goal.view(-1, self.state_dim))
        # Hand losses
        self.left_loss = self._hand_loss(beam_poses, left_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        self.right_loss = self._hand_loss(beam_poses, right_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        
        # Calculate gradients
        beam_gradients = grad(outputs=beam_losses, inputs=beam_poses, retain_graph=True)[0]
        
        # Check if any of the goals have converged
        self.check_convergence(beam_gradients)        
        
        # Use loss to calculate contacts
        left_contacts = (self.left_loss < tol).type(torch.float32)
        right_contacts = (self.right_loss < tol).type(torch.float32)
        
        self.left_loss[pin_indices] *= 10.0
        self.right_loss[pin_indices] *= 10.0
        
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
        
        return torch.cat([left_gradients, right_gradients, beam_gradients.view(-1)], dim=0), beam_losses
    
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
    
    def set_x(self, left_hand, right_hand, beams):
        self._x = torch.cat([left_hand, right_hand, beams], dim=0)
        return 