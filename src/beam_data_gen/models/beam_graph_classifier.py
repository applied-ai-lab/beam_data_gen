import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.classifier import Classifier 

from beam_data_gen.models.beam_vae_params import BeamVaeParams


class BeamGraphClassifier(Classifier):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        self._loss = nn.CrossEntropyLoss()
        
        # Classifier weights
        self._loss_weights = torch.tensor([[0., 1., 1.], 
                                           [0., 0., 1.], 
                                           [0., 0., 0.]]).to(self.vae_params.device)
        
    def forward(self, inputs: torch.tensor):
        logits = self.class_mlp(inputs)
        # Reshape to batch_size, no_nodes, no_nodes
        logits_reshaped = logits.reshape(inputs.shape[0], 
                                         self.vae_params.no_classifier_nodes, 
                                         self.vae_params.no_classifier_nodes)
        edge_pred = 0.5 * (logits_reshaped.transpose(1, 2) + logits_reshaped)
        return edge_pred
    
    def loss_func(self, edge_logits: torch.tensor, edge_targets: torch.tensor):    
        return F.binary_cross_entropy_with_logits(edge_logits, edge_targets, reduction="sum", weight=self._loss_weights)
    
    @property
    def loss_weights(self):
        return self._loss_weights
    
