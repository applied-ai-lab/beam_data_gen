import torch
from torch import nn
from torch.nn import functional as F

from vae_planner.models.encoder_base import EncoderBase, VaeInputsBase

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class BeamEncoder(EncoderBase):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        # Define encoder MLP
        self.beam_1 = nn.Sequential(

            nn.Linear(self.vae_params.input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, 2*self.vae_params.beam_latent_dim),

        )
        
        self.beam_2 = nn.Sequential(

            nn.Linear(self.vae_params.input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, 2*self.vae_params.beam_latent_dim),

        )
        
        self.pin_A = nn.Sequential(

            nn.Linear(self.vae_params.input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, 2*self.vae_params.beam_latent_dim),

        )
    
    def forward(self, inputs: VaeInputsBase) -> tuple:
        xin_beam_1 = inputs.x_in[:, 0:self.vae_params.state_dim]
        xin_beam_2 = inputs.x_in[:, self.vae_params.state_dim: 2*self.vae_params.state_dim]
        xin_pin_A = inputs.x_in[:, 2*self.vae_params.state_dim: 3*self.vae_params.state_dim]
        
        out_beam_1 = self._enc_forward(xin_beam_1, self.beam_1)
        out_beam_2 = self._enc_forward(xin_beam_2, self.beam_2)
        out_pin_A  = self._enc_forward(xin_pin_A, self.pin_A)
        
        self._latents.z = torch.cat([out_beam_1[0], out_beam_2[0], out_pin_A[0]], dim=1)
        self._latents.mu = torch.cat([out_beam_1[1], out_beam_2[1], out_pin_A[1]], dim=1)
        self._latents.log_var = torch.cat([out_beam_1[2], out_beam_2[2], out_pin_A[2]], dim=1)

        return self._latents
    
    def _enc_forward(self, x_in: torch.tensor, mlp: nn.Sequential):
        mlp_out = mlp(x_in)
        # Take the top half as the mean
        mu = mlp_out[:, 0:self.vae_params.beam_latent_dim]
        # Take the bottom half as the variance
        var = F.softplus(mlp_out[:, self.vae_params.beam_latent_dim:]) + 1e-5
        std = torch.sqrt(var)
        # Sample eps from standard Gaussian of size of std
        eps = torch.randn_like(std)

        # Use the logVar for the loss
        log_var = torch.log(var)
        
        # Sample z
        z = mu + eps * std
        return mu, log_var, z
        
        
