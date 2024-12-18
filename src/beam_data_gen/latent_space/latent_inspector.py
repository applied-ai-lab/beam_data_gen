import torch
import numpy as np

from matplotlib import pyplot as plt
from vae_planner.latent_space.latent_inspector import LatentInspector

from beam_data_gen.models.beam_vae_pp import BeamVae, BeamVaeParams, BeamVaeInputs


class BeamLSInspector(LatentInspector):
        
    def __init__(self, model: BeamVae, vae_params: BeamVaeParams) -> None:
        super().__init__(model, vae_params)
        
        self.colour_code = {
            str([0, 0]): {'colour': '#EE6677', 
                           'label': 'Unconnected',
                         'plotted': False}, 
            str([0, 1]): {'colour': '#3E9ABB', 
                           'label': 'Undefined',
                         'plotted': False},
            str([1, 0]): {'colour': '#0077BB', 
                           'label': 'Beams assembled',
                         'plotted': False},
            str([1, 1]): {'colour': '#EE7733', 
                           'label': 'Complete',
                         'plotted': False},        
        }
        
    def find_latent_dims(self, inputs: BeamVaeInputs):
        with torch.no_grad():
            # Encode
            latents = self.model.encoder(inputs)

            # Mean of the Log var
            mean_var = latents.log_var.exp().mean(0).cpu().numpy()

            latent_dims = np.argsort(mean_var)
            return latent_dims, mean_var[latent_dims]
        
    def plot_latents(self, x: torch.tensor, y: torch.tensor, batched_graphs: torch.tensor):
        # Convert values to numpy arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        graphs_np = batched_graphs.detach().cpu().detach().numpy()
        
        # Create colours        
        colour_list = list(self.colour_code[str([graphs_np[k, 0, 1].astype(int), 
                                                 graphs_np[k, 0, 2].astype(int)])] for k in range(graphs_np.shape[0]))
        
        fig, axis = plt.subplots(1, 1)
        axis.axis('equal')
        for k in range(x_np.shape[0]):
            if not colour_list[k]["plotted"]:
                axis.scatter(x_np[k], y_np[k], color=colour_list[k]["colour"], label=colour_list[k]["label"])
                colour_list[k]["plotted"] = True
            else:
                axis.scatter(x_np[k], y_np[k], color=colour_list[k]["colour"])
        axis.legend()
        return axis
