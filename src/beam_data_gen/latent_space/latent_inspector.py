import torch
import numpy as np

from vae_planner.latent_space.latent_inspector import LatentInspector

from beam_data_gen.models.beam_vae_pp import BeamVae, BeamVaeParams, BeamVaeInputs


class BeamLSInspector(LatentInspector):
    def __init__(self, model: BeamVae, vae_params: BeamVaeParams) -> None:
        super().__init__(model, vae_params)
        
    def find_latent_dims(self, inputs: BeamVaeInputs):
        with torch.no_grad():
            # Encode
            latents = self.model.encoder(inputs)

            # Mean of the Log var
            mean_var = latents.log_var.exp().mean(0).cpu().numpy()

            latent_dims = np.argsort(mean_var)
            return latent_dims, mean_var[latent_dims]
