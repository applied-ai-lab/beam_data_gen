from torch import nn

from vae_planner.parameter_handlers.vae_params import VaeParams
from vae_planner.models.decoder_base import DecoderBase


class BeamDecoder(DecoderBase):
    def __init__(self,
                 vae_params: VaeParams):
        super().__init__(vae_params)
        
        # Define decoder MLP
        self.decoder_mlp = nn.Sequential(

            nn.Linear(self.vae_params.decoder_input_dim, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.model_width),
            nn.ELU(),
            nn.Linear(self.vae_params.model_width, self.vae_params.output_dim),

        )