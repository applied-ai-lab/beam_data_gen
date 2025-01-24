from typing import Dict
import copy

import torch
import numpy as np
from matplotlib import pyplot as plt

from beam_data_gen.latent_space.latent_inspector import BeamLSInspector
from beam_data_gen.models.beam_containers import LatentVarsBase
from beam_data_gen.models.beam_robot_vae import BeamRobotVae, BeamVaeParams, BeamRobotInputs, BeamRobotLatents


class RobotBeamLSInspector(BeamLSInspector):
        
    def __init__(self, model: BeamRobotVae, vae_params: BeamVaeParams) -> None:
        super().__init__(model, vae_params)
        
        
    def find_latent_dims(self, inputs: BeamRobotInputs) -> Dict[str, tuple]:
        with torch.no_grad():
            # Encode
            latents: BeamRobotLatents = self.model.encoder(inputs)
            
            def find_dims(latent_vars):
                # Mean of the Log var
                mean_var = latent_vars.log_var.exp().mean(0).cpu().numpy()
                latent_dims = np.argsort(mean_var)
                return latent_dims, mean_var[latent_dims]
            
            latent_dims_dict = {'robot': copy.deepcopy(find_dims(latents.robot)),
                                'beams': copy.deepcopy(find_dims(latents.beams))}
            
            return latent_dims_dict
        
    def sample_latent_space(self, radius, latent_dict: Dict[str, tuple], no_samps: int) -> BeamRobotLatents:
        
        x, y = self.sample_latent_values_from_unit_circle_2d(radius, no_samps)
        
        latents_for_plotting = BeamRobotLatents()
        z_dict = {}
        z_dict['robot'] = torch.zeros([no_samps, self.vae_params.robot_latent_dim], dtype=torch.float32).to(self.vae_params.device)
        z_dict['beams'] = torch.zeros([no_samps, self.vae_params.beam_latent_dim], dtype=torch.float32).to(self.vae_params.device)
        
        for key, item in latent_dict.items():
            if key == 'robot':
                z_dict[key][:, item[0][0]] = copy.deepcopy(x)
                z_dict[key][:, item[0][1]] = copy.deepcopy(y)
            else:
                z_dict[key][:, item[0][0]] = copy.deepcopy(x)
                z_dict[key][:, item[0][1]] = copy.deepcopy(y)
            
            z_var = LatentVarsBase()
            z_var.z = z_dict[key]
            
            setattr(latents_for_plotting, key, z_var)
        
        return latents_for_plotting
