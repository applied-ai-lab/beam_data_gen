import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.classifier import Classifier 

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class Indices:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class IndependentClassifier(Classifier):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        ## Sizes and indices
        self._no_predictions = self.find_no_non_zero_values(self.vae_params.no_classifier_nodes)
        
        self.triu_indices = torch.triu_indices(self.vae_params.no_classifier_nodes, self.vae_params.no_classifier_nodes, offset=1)
        
        assert vae_params.latent_dim % self._no_predictions == 0, \
            f"{vae_params.latent_dim} is not exactly divisible by {self._no_predictions}"
        
        ## Latent indices
        self._independent_latent_dim = self.vae_params.latent_dim // self._no_predictions
        self._latent_indices = self.create_latent_indices()        
        
        ## Classifiers
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._independent_latent_dim, self.vae_params.model_width),
                nn.ELU(),
                nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
                nn.ELU(),
                nn.Linear(self.vae_params.model_width, 1))
            for _ in range(self._no_predictions)
        ])        
        
        ## Loss objects
        self._loss = nn.CrossEntropyLoss()
        
        # Classifier weights
        self._loss_weights = (torch.ones([self.vae_params.no_classifier_nodes, 
                                          self.vae_params.no_classifier_nodes], dtype=torch.float32)
                              - 
                              torch.eye(self.vae_params.no_classifier_nodes)
                              ).to(self.vae_params.device)
        
        self._loss_crit = nn.BCEWithLogitsLoss(weight=self._loss_weights, reduction="sum")
        
    def forward(self, inputs: torch.tensor):
                
        out = torch.cat([clf(inputs[:, self._latent_indices[k].start: self._latent_indices[k].end]) for k, clf in enumerate(self.classifiers)], dim=1)
    
        logits_reshaped = torch.zeros([inputs.shape[0], 
                                    self.vae_params.no_classifier_nodes,
                                    self.vae_params.no_classifier_nodes], 
                                    dtype=torch.float32).to(self.vae_params.device)
        
        logits_reshaped[:, self.triu_indices[0], self.triu_indices[1]] = out
        
        edge_pred = 0.5 * (logits_reshaped.transpose(1, 2) + logits_reshaped)
        edge_pred = edge_pred * (1 - torch.eye(edge_pred.shape[1], edge_pred.shape[1]).repeat(edge_pred.shape[0], 1, 1)).to(self.vae_params.device)
        return edge_pred
    
    def loss_func(self, edge_logits: torch.tensor, edge_targets: torch.tensor):    
        return self._loss_crit(edge_logits, edge_targets)
    
    ## Private methods
    def find_no_non_zero_values(self, no_nodes):
        num = 0
        i = no_nodes - 1
        while i > 0:
            num += i
            i -= 1
        return num
    
    def create_latent_indices(self):
        index_dict = {}
        latent_dim = self.vae_params.latent_dim // self._no_predictions
        for k in range(self._no_predictions):
            index_dict[k] = Indices(k * latent_dim, (k + 1) * latent_dim)
        return index_dict
    
    ## Getters and Setters
    @property
    def loss_weights(self):
        return self._loss_weights
    
    @property
    def no_predictions(self):
        self._no_predictions = self.find_no_non_zero_values(self.vae_params.no_classifier_nodes)
        return self._no_predictions
