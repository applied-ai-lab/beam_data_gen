import torch
from torch import nn
from torch.nn import functional as F

from beam_data_gen.models.classifiers.beam_graph_classifier import BeamGraphClassifier
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class GraphClassifier(BeamGraphClassifier):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        # Define encoder MLP
        self.class_mlp = nn.Sequential(

            nn.Linear(self.vae_params.latent_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.no_classifier_pred)

        )