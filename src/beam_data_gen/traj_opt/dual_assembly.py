import torch
from torch import nn
from torch.autograd import grad
import numpy as np
import mujoco
from filterpy.monte_carlo import systematic_resample

from beam_data_gen.traj_opt.traj_opt_base import (TrajOptParams, TrajOptBase)
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim


class DualAssembly(TrajOptBase):
    def __init__(self, params: TrajOptParams, state_dim: int, sim: SquareRobotSim):
        super().__init__(params)
        self.state_dim = state_dim
        
        self.sim = sim
        
        self._left_index = -1
        self._right_index = -1
        
        # Quantities
        self._counter = 0
        self._no_hands = 2
        # Losses
        self._beam_loss = nn.MSELoss(reduction="sum")
        self._hand_loss = nn.MSELoss(reduction='none')    
    
    def optimise(self, model, data):
        
        self._counter = 0
        particles = torch.zeros([self.params.no_particles, self.params.no_steps + 1, self._x.shape[0]], dtype=self._x.dtype).to(self._x.device)
        weights = torch.zeros([self.params.no_particles], dtype=self._x.dtype).to(self._x.device)
        
        for k in range(self._params.no_steps):
            for n in range(self._params.no_particles):
            
                x_grads = self._gradients()
                
                # Apply noise to gradients
                noise = 1.0 / float(k + 1) * torch.randn_like(x_grads)
                self._x = self._x - self.params.step_size * (x_grads + 0.1 * noise)
                
                self.normalise_pose(self._x)
                particles[n, k, :] = self._x
                
                # Check collisions
                self.sim.decode_x(data, self._x)
                mujoco.mj_step(model, data)
                
                # Check for collisions with moving beams
                if self.sim.check_collisions(data, self.node_names[self._left_index]) or self.sim.check_collisions(data, self.node_names[self._right_index]):
                    weights[n] = 1.0e-5
                else:
                    weights[n] = 1.0
            
            # Resample particles
            weights += 1e-6
            weights /= weights.sum()
            
            indices = systematic_resample(weights)
            particles[:, k, :] = particles[indices, k, :]
        
        return particles
    
    
    def _gradients(self, tol: float):
        left_hand = self._x[0:self.state_dim]
        right_hand = self._x[self.state_dim: 2 * self.state_dim]
        beam_poses = self._x[self._no_hands * self.state_dim: ].view(-1, self.state_dim)
        # Calculate losses
        beam_losses = self._beam_loss(beam_poses, self.goal.view(-1, self.state_dim))
        # Hand losses
        left_loss = self._hand_loss(beam_poses, left_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        right_loss = self._hand_loss(beam_poses, right_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
        
        # Use loss to calculate contacts
        left_contacts = (left_loss < tol).type(torch.float32)
        right_contacts = (right_loss < tol).type(torch.float32)
        
        # Calculate gradients
        beam_gradients = grad(outputs=beam_losses, inputs=beam_poses, retain_graph=True)[0]
        
        # Check if any of the goals have converged
        index = self.check_convergence(beam_gradients, self._counter)
        
        # Update hand losses to ignore assembling these
        left_loss[index] *= 1.0e6
        right_loss[index] *= 1.0e6
        
        # Find smallest gradients
        self._right_index = torch.argmin(right_loss, 0)
        self._left_index = torch.argmin(left_loss, 0)
        
        # Figure out what to do if a beam is equi-distant
        if self._right_index == self._left_index:
            right_loss[self._right_index] = 1.0e3
            self._right_index = torch.argmin(right_loss, 0)
            
        beam_gradients = (beam_gradients * left_contacts.reshape(beam_gradients.shape[0], 1) + beam_gradients * right_contacts.reshape(beam_gradients.shape[0], 1))
    
        # Hand gradients
        left_gradients = grad(left_loss[self._left_index], inputs=left_hand, retain_graph=True)[0] * (1. - left_contacts[self._left_index]) + beam_gradients[self._left_index, :]
        right_gradients = grad(right_loss[self._right_index], inputs=right_hand, retain_graph=True)[0] * (1. - right_contacts[self._right_index]) + beam_gradients[self._right_index, :]
        
        return torch.cat([left_gradients, right_gradients, beam_gradients], dim=0)
    
    def check_convergence(self, gradient, counter):
        gradient_reshaped = gradient.view(-1, self.state_dim)
        grad_norm = torch.norm(gradient_reshaped, p=2.0, dim=1)
        # Item with largest gradient
        index = torch.argmin(grad_norm)
        min_val = grad_norm[index]
        
        while min_val < 0.01:
            
            counter += 1
            if counter >= gradient_reshaped.shape[0]:
                gradient *= 0.0
                break
            
            gradient_reshaped[index, :] *= 1.0e6      
            
            grad_norm = torch.norm(gradient_reshaped, p=2.0, dim=1)
            index = torch.argmin(grad_norm)
            min_val = grad_norm[index]
            
        return index
    
    def normalise_pose(self, pose_torch: torch.tensor):
        no_items = pose_torch.shape[0] // self.state_dim    
        for k in range(no_items):
            pose_torch[self.state_dim * k + 3: self.state_dim * k + 5] = torch.nn.functional.normalize(pose_torch[self.state_dim * k + 3: self.state_dim * k + 5], dim=0)
        return pose_torch
    
    @x.setter
    def x(self, left_hand, right_hand, beams):
        self._x = torch.cat([left_hand, right_hand, beams], dim=0)
        return