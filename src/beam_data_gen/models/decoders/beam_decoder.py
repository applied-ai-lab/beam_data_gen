import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.parameter_handlers.vae_params import VaeParams
from vae_planner.models.container_base import (VaeOutputsBase, LatentVarsBase)
from vae_planner.models.decoder_base import DecoderBase


class BeamDecoder(DecoderBase):
    def __init__(self,
                 vae_params: VaeParams):
        super().__init__(vae_params)
    
        
    def forward(self, latents: LatentVarsBase, action: torch.tensor = None):
        x_out = self.decoder_mlp(latents.z)

        self._vae_outputs.x_pred = x_out
        return self._vae_outputs
    