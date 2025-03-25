import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.container_base import (VaeOutputsBase, LatentVarsBase)
from vae_planner.models.decoder_base import DecoderBase

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams
from beam_data_gen.models.containers.beam_robot_containers import BeamRobotLatents


class BeamRobotDecoder(DecoderBase):
    def __init__(self,
                 vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        # Define encoder MLP
        self.beams = nn.Sequential(

            nn.Linear(self.vae_params.beam_latent_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.beam_output_dim),

        )
        
        self.hands = nn.Sequential(

            nn.Linear(self.vae_params.robot_latent_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.robot_output_dim),

        )
    
        
    def forward(self, latents: BeamRobotLatents, action: torch.tensor = None):
        x_robot_out = self.hands(latents.robot.z)
        x_beams_out = self.beams(latents.beams.z)

        self._vae_outputs.x_pred = torch.cat([x_robot_out, x_beams_out], dim=1)
        return self._vae_outputs
    