import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.classifier import Classifier 

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class LinearClassifier(Classifier):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        self._loss = nn.CrossEntropyLoss()
        
        # Define encoder MLP
        self.class_mlp = nn.Sequential(

            nn.Linear(self.vae_params.latent_dim, self.vae_params.no_classifier_pred)

        )
        
        # Classifier weights
        self._loss_weights = (torch.ones([self.vae_params.no_classifier_nodes, 
                                          self.vae_params.no_classifier_nodes], dtype=torch.float32)
                              - 
                              torch.eye(self.vae_params.no_classifier_nodes)
                              ).to(self.vae_params.device)
        
        self._loss_crit = nn.BCEWithLogitsLoss(weight=self._loss_weights, reduction="sum")
        
    def forward(self, inputs: torch.tensor):
        logits = self.class_mlp(inputs)
        # Reshape to batch_size, no_nodes, no_nodes
        logits_reshaped = logits.reshape(inputs.shape[0], 
                                         self.vae_params.no_classifier_nodes, 
                                         self.vae_params.no_classifier_nodes)
        edge_pred = 0.5 * (logits_reshaped.transpose(1, 2) + logits_reshaped)
        edge_pred = edge_pred * (1 - torch.eye(edge_pred.shape[1], edge_pred.shape[1]).repeat(edge_pred.shape[0], 1, 1)).to(self.vae_params.device)
        return edge_pred
    
    def loss_func(self, edge_logits: torch.tensor, edge_targets: torch.tensor):    
        return self._loss_crit(edge_logits, edge_targets)
    
    @property
    def loss_weights(self):
        return self._loss_weights
    
