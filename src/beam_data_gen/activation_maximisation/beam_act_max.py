from typing import List
import copy

import torch
from torch import nn
from torch.autograd import grad

from beam_data_gen.models.beam_vae_pp import BeamVae, LatentVarsBase

from vae_planner.activation_maximisation.act_max import (ActivationMaximisation, ActMaxOutput, ActMaxParams)


class BeamSetPoint:
    def __init__(self, graph_target: torch.tensor, no_iters: int):
        self.graph_target = torch.tensor(graph_target, dtype=torch.float32)
        self.no_iters = no_iters
        

class BeamActMax(ActivationMaximisation):
    def __init__(self, params, device):
        super().__init__(params, device)
        
    def optimise(self, model: BeamVae, z0: torch.tensor, set_points: List[BeamSetPoint]) -> torch.tensor:
        
        assert len(z0.shape) == 1, "z0 must be a 1D vector"
        
        # Number of x features
        x_features = model.vae_params.output_dim
        
        # Find traj len
        traj_len = 1
        for set_point in set_points:
            traj_len += set_point.no_iters
        
        # Traj_len x features -- make a leaf node
        z_traj = torch.zeros([traj_len, z0.shape[0]], dtype=torch.float32).to(self._device)
        # Set init condition
        z_traj[0, :] = z0.clone()
        z_traj.requires_grad_(True)
        # Init graph
        graph0 = model._classifier.forward(z0.unsqueeze(0))
        
        # Graph targets
        graph_targets = torch.zeros([len(set_points), 
                                    model.vae_params.no_classifier_nodes, 
                                    model.vae_params.no_classifier_nodes], dtype=torch.float32).to(self._device)
        graph_targets[0, :, :] = graph0.clone()
        
        k = 0
        graph_indices = []
        cumaltive_iters = 0
        for i, set_point in enumerate(set_points):
            cumaltive_iters += set_point.no_iters
            graph_targets[i, :, :] = set_point.graph_target.to(self._device)
            graph_indices.append(cumaltive_iters)
            # for _ in range(set_point.no_iters):
            #     graph_targets[k + 1, :, :] = set_point.graph_target.to(self._device)
            #     # Iterate counter
            #     k += 1
                
        # Create smoothness loss to penalize jerk
        # A_mat = torch.matmul(self.vel_constraint(traj_len - 2, x_features),
        #                     torch.matmul(self.vel_constraint(traj_len - 1, x_features),
        #                     self.vel_constraint(traj_len, x_features)))     
        A_mat = self.vel_constraint(traj_len, x_features)
        x_vel_tar = torch.zeros((x_features * (traj_len - 1), 1), dtype=torch.float32).to(self._device)
        
        latents = LatentVarsBase()
        latents.z = z_traj
        
        self._counter = 0
        self._loss[0] = 1000
        
        mse_loss = nn.MSELoss(reduction='sum')
        graph_loss = nn.BCEWithLogitsLoss()       
        
        tot_loss = torch.zeros([self._params.max_iters], dtype=torch.float32).to(self._device)       
        
        while self._counter < self._params.max_iters and torch.norm(self._loss) > self._params.stop_criterion:
            
            graph_pred = model._classifier.forward(latents.z)
            x_out = model.decoder(latents, None)
            # self._loss = graph_loss(graph_pred, graph_targets) + \
            #                 0.1 * mse_loss(torch.matmul(A_mat, x_out.x_pred.reshape(-1, 1)), x_vel_tar)
            self._loss = graph_loss(graph_pred[graph_indices, :, :], graph_targets) + \
                            10.0 * mse_loss(torch.matmul(A_mat, x_out.x_pred.reshape(-1, 1)), x_vel_tar)
                            
            grad_features = grad(outputs=self._loss, inputs=latents.z, retain_graph=True)[0]
            grad_features[0, :] *= 0.0
            
            latents.z = latents.z - self._params.lr * grad_features
            
            tot_loss[self._counter] = self._loss
            self._counter += 1
        
        return ActMaxOutput(latents.z, tot_loss, copy.deepcopy(self._counter))
    
    def optimise_primal(self, model: BeamVae, latents: LatentVarsBase, graph_targets: torch.tensor):
        
        traj_len = latents.z.shape[0]
        # Predict graph features
        graph_pred = model._classifier.forward(latents.z)
        
        # Number of x features
        # no_features = model.vae_params.output_dim
        no_features = latents.z.shape[1]
        
        # Smoothness in LS loss
        A_mat = self.vel_constraint(latents.z.shape[0], latents.z.shape[1])
        # A_mat = torch.matmul(self.vel_constraint(traj_len - 2, no_features),
        #                     torch.matmul(self.vel_constraint(traj_len - 1, no_features),
        #                     self.vel_constraint(traj_len, no_features)))     
        vel_tar = torch.zeros([A_mat.shape[0], 1], dtype=torch.float32).to(self._device)
        
        self._counter = 0
        self._loss[0] = 1000
        
        mse_loss = nn.MSELoss(reduction='sum')
        graph_loss = nn.BCEWithLogitsLoss()       
        
        tot_loss = torch.zeros([self._params.max_iters], dtype=torch.float32).to(self._device)       
        
        while self._counter < self._params.max_iters and torch.norm(self._loss) > self._params.stop_criterion:
            
            graph_pred = model._classifier.forward(latents.z)
            x_out = model.decoder(latents, None)
            self._loss = 0.01 * graph_loss(graph_pred, graph_targets) + \
                            0.1 * mse_loss(torch.matmul(A_mat, latents.z.reshape(-1, 1)), vel_tar)
                            
            grad_features = grad(outputs=self._loss, inputs=latents.z, retain_graph=True)[0]
            grad_features[0, :] *= 0.0
            grad_features[-1, :] *= 0.0
            
            latents.z = latents.z - self._params.lr * grad_features
            
            tot_loss[self._counter] = self._loss
            self._counter += 1
        
        return ActMaxOutput(latents.z, tot_loss, copy.deepcopy(self._counter))         
        
    # Helper functions
    
    def vel_constraint(self, traj_len: int, no_features: int):
        # Create smoothness loss
        A0 = - torch.cat([torch.eye((traj_len - 1) * no_features), 
                        torch.zeros([(traj_len - 1) * no_features, no_features])], dim=1).to(self._device)
        
        A1 = torch.cat([torch.zeros([(traj_len - 1) * no_features, no_features]), 
                        torch.eye((traj_len - 1) * no_features)], dim=1).to(self._device)
        return A0 + A1
