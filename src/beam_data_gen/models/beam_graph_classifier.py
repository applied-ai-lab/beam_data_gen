import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.classifier import Classifier 

from beam_data_gen.models.beam_vae_params import BeamVaeParams


class BeamGraphClassifier(Classifier):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        self._loss = nn.CrossEntropyLoss()
        
    def forward(self, inputs: torch.tensor):
        logits = self.class_mlp(inputs)
        # Reshape to batch_size, no_nodes, no_nodes
        logits_reshaped = logits.reshape(inputs.shape[0], 
                                         self.vae_params.no_classifier_nodes, 
                                         self.vae_params.no_classifier_nodes)
        edge_pred = 0.5 * (logits_reshaped.transpose(1, 2) + logits_reshaped)
        return edge_pred
    
    def loss_func(self, edge_logits: torch.tensor, edge_targets: torch.tensor):
        target_up = torch.triu(edge_targets, diagonal=1)
        output_up = torch.triu(edge_logits, diagonal=1)      
                
        return self._loss(output_up, target_up)
    
