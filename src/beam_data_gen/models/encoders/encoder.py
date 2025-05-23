from torch import nn

from vae_planner.models.encoder_base import EncoderBase

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


class Encoder(EncoderBase):
    def __init__(self, vae_params: BeamVaeParams):
        super().__init__(vae_params)
        
        # Define encoder MLP
        self.encoder_mlp = nn.Sequential(

            nn.Linear(self.vae_params.input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, 2*self.vae_params.latent_dim),

        )