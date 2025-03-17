import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.encoder_base import EncoderBase

from beam_data_gen.models.containers.beam_robot_containers import (BeamRobotInputs, BeamRobotLatents, BeamRobotOutputs)
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class BeamRobotEncoder(EncoderBase):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        # Define encoder MLP
        self.beams = nn.Sequential(

            nn.Linear(self.vae_params.beam_input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, 2*self.vae_params.beam_latent_dim),

        )
        
        self.hands = nn.Sequential(

            nn.Linear(self.vae_params.robot_input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, 2*self.vae_params.robot_latent_dim),

        )
        
        self._latents = BeamRobotLatents()
    
    def forward(self, inputs: BeamRobotInputs) -> tuple:
        x_robot_in = inputs.x_in[:, 0:self.vae_params.robot_input_dim]
        x_beam_in = inputs.x_in[:, self.vae_params.robot_input_dim: self.vae_params.robot_input_dim + self.vae_params.beam_input_dim]
        
        self._latents.beams.mu, self._latents.beams.log_var, self._latents.beams.z = self._enc_forward(x_beam_in, self.beams)
        self._latents.robot.mu, self._latents.robot.log_var, self._latents.robot.z = self._enc_forward(x_robot_in, self.hands)        
        # Latents concatenates the variables above
        return self._latents
    
    def _enc_forward(self, x_in: torch.tensor, mlp: nn.Sequential):
        mlp_out = mlp(x_in)
        # Take the top half as the mean
        mu = mlp_out[:, 0:int(mlp_out.shape[1] / 2)]
        # Take the bottom half as the variance
        var = F.softplus(mlp_out[:, int(mlp_out.shape[1] / 2):]) + 1e-5
        std = torch.sqrt(var)
        # Sample eps from standard Gaussian of size of std
        eps = torch.randn_like(std)

        # Use the logVar for the loss
        log_var = torch.log(var)
        
        # Sample z
        z = mu + eps * std
        return mu, log_var, z
        
        
