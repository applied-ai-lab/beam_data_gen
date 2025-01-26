import torch
import numpy as np

from matplotlib import pyplot as plt
from vae_planner.latent_space.latent_inspector import LatentInspector

from beam_data_gen.models.beam_vae_pp import BeamVae, BeamVaeParams, BeamVaeInputs
from beam_data_gen.beam_impl.robot_graph import RampGraph, RobotGraph
from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected)


class BeamLSInspector(LatentInspector):
        
    def __init__(self, model: BeamVae, vae_params: BeamVaeParams) -> None:
        super().__init__(model, vae_params)
        
        self.beam_only_colours = {
            self.graph_to_key(l_disconnected): {'colour': '#EE6677', 
                           'label': 'Disconnected',
                         'plotted': False}, 
            'default': {'colour': '#3E9ABB', 
                           'label': 'Undefined',
                         'plotted': False},
            self.graph_to_key(l_pin_removed): {'colour': '#0077BB', 
                           'label': 'Pin removed',
                         'plotted': False},
            self.graph_to_key(l_connected_graph): {'colour': '#EE7733', 
                           'label': 'Complete',
                         'plotted': False},        
        }

    def graph_to_key(self, ramp_graph: RampGraph):
        return self.adj_mat_to_key(ramp_graph.A)
    
    def adj_mat_to_key(self, A: np.array):
        a_list = A.astype(int).flatten().tolist()
        return ' '.join(map(str, a_list))
    
        
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
        if batched_graphs.shape[1] == 3:  
            colour_list = list(self.beam_only_colours[self.adj_mat_to_key(graphs_np[k, :, :])] for k in range(graphs_np.shape[0]))
        elif batched_graphs.shape[1] == 5:  
            colour_list = list(self.beam_only_colours[self.adj_mat_to_key(graphs_np[k, :, :])] for k in range(graphs_np.shape[0]))
            return
        
        
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
