import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.classifier import Classifier 

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class SpaceClassifier(Classifier):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        self._loss = nn.CrossEntropyLoss()
        
        # Define encoder MLP
        self.graph_mlp = nn.Sequential(

            nn.Linear(self.vae_params.robot_latent_dim , self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.no_classifier_pred)

        )
        
        # Free space MLP
        self.space_mlp = nn.Sequential(

            nn.Linear(self.vae_params.beam_latent_dim , self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.no_hands)

        )
        
        self._loss_crit = nn.BCEWithLogitsLoss(reduction="sum")
        
    def graph_forward(self, inputs: torch.tensor):
        logits = self.graph_mlp(inputs)
        # Reshape to batch_size, no_nodes, no_nodes
        logits_reshaped = logits.reshape(inputs.shape[0], 
                                         self.vae_params.no_classifier_nodes, 
                                         self.vae_params.no_classifier_nodes)
        edge_pred = 0.5 * (logits_reshaped.transpose(1, 2) + logits_reshaped)
        edge_pred = edge_pred * (1 - torch.eye(edge_pred.shape[1], edge_pred.shape[1]).repeat(edge_pred.shape[0], 1, 1)).to(self.vae_params.device)
        return edge_pred
    
    def space_forward(self, inputs: torch.tensor):
        logits = self.space_mlp(inputs)
        return logits
    
    def forward(self, inputs: torch.tensor):
        # Forward through models
        graph_pred = self.graph_forward(inputs[:, 0:self.vae_params.robot_latent_dim])
        space_pred = self.space_mlp(inputs[:, self.vae_params.robot_latent_dim: 
                                                self.vae_params.robot_latent_dim + self.vae_params.beam_latent_dim])
        
        non_diag_values = self.get_offdiagonals(graph_pred)
        
        return torch.cat([non_diag_values, space_pred], dim=1)
    
    def loss_func(self, edge_logits: torch.tensor, edge_targets: torch.tensor):    
        return self._loss_crit(edge_logits, edge_targets)
    
    def get_offdiagonals(self, input: torch.tensor):
        mask = ~torch.eye(input.shape[1], dtype=torch.bool, device=input.device)
        return input[:, mask]
    
