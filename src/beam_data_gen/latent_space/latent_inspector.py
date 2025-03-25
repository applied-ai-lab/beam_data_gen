import copy

import torch
import numpy as np

from matplotlib import pyplot as plt
from vae_planner.latent_space.latent_inspector import LatentInspector

from beam_data_gen.models.vaes.beam_vae_pp import BeamVae, BeamVaeParams, BeamVaeInputs
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
        
        self.hand_connections = {
            ' '.join(map(str, [0, 0, 0, 0, 0])) : {'colour': 'r', 
                            'label': 'Free space',
                            'plotted': False},
            ' '.join(map(str, [1, 0, 0, 0, 0])) : {'colour': '#0077BB', 
                            'label': 'Left hand',
                            'plotted': False},
            ' '.join(map(str, [0, 1, 0, 0, 0])) : {'colour': '#EE7733', 
                            'label': 'Right hand',
                            'plotted': False},
            ' '.join(map(str, [0, 0, 1, 0, 0])) : {'colour': 'c', 
                            'label': 'Beam 1',
                            'plotted': False},
            ' '.join(map(str, [0, 0, 0, 1, 0])) : {'colour': 'm', 
                            'label': 'Beam 2',
                            'plotted': False},
            ' '.join(map(str, [0, 0, 0, 0, 1])) : {'colour': 'g', 
                            'label': 'Pin',
                            'plotted': False}
        }
        
    def reset_colour_dict(self, colour_list):
        for c_dict in colour_list:
            c_dict['plotted'] = False
        return    

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
        
    def split_latent_dims(self, latent_dims):
        if not self.vae_params.split_encoder:
            return latent_dims
        else:
            latents_robot = []
            latents_beams = []
            
            for k in range(latent_dims.shape[0]):
                if latent_dims[k] < self.vae_params.robot_latent_dim:
                    latents_robot.append(latent_dims[k])
                else:
                    latents_beams.append(latent_dims[k])
            return (latents_robot, latents_beams)                
        
    def plot_latents(self, x: torch.tensor, y: torch.tensor, batched_graphs: torch.tensor, title: str=None):
        # Convert values to numpy arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        graphs_np = batched_graphs.detach().cpu().detach().numpy()
        
        
        # Create colours   
        # beam_list = list(self.beam_only_colours[self.adj_mat_to_key(graphs_np[k, -3:, -3:])] for k in range(graphs_np.shape[0]))
        
        undefined_dict = {'colour': 'k', 
                            'label': 'Undefined',
                            'plotted': False}
        
        beam_list = []
        for k in range(graphs_np.shape[0]):
            try:
                beam_list.append(self.beam_only_colours[self.adj_mat_to_key(graphs_np[k, -3:, -3:])])
            except KeyError:
                beam_list.append(undefined_dict)
            
        left_list = []
        for k in range(graphs_np.shape[0]):
            try:
                left_list.append(self.hand_connections[self.adj_mat_to_key(graphs_np[k, 0, :])])
            except KeyError:
                left_list.append(undefined_dict)
                
        right_list = []
        for k in range(graphs_np.shape[0]):
            try:
                right_list.append(self.hand_connections[self.adj_mat_to_key(graphs_np[k, 1, :])])
            except:
                right_list.append(undefined_dict)
        

        colour_list = [beam_list, left_list, right_list]
        
        fig, axes = plt.subplots(1, 3)
        fig.suptitle(title)
        for axis_counter, axis in enumerate(axes):
            axis.axis('equal')
            for k in range(x_np.shape[0]):
                if not colour_list[axis_counter][k]["plotted"]:
                    axis.scatter(x_np[k], y_np[k], color=colour_list[axis_counter][k]["colour"], label=colour_list[axis_counter][k]["label"])
                    colour_list[axis_counter][k]["plotted"] = True
                else:
                    axis.scatter(x_np[k], y_np[k], color=colour_list[axis_counter][k]["colour"])
            axis.legend()
        
        for c_list in colour_list:
            self.reset_colour_dict(c_list)
        
        return fig, axes
    
    def plot_freespace_latents(self, 
                               x: torch.tensor, 
                               y: torch.tensor, 
                               batched_graphs: torch.tensor, 
                               left_free_space: torch.tensor,
                               title: str=None):
        # Convert values to numpy arrays
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        graphs_np = batched_graphs.detach().cpu().detach().numpy()
        
        free_space_np = left_free_space.detach().cpu().detach().numpy()
        
        
        # Create colours   
        # beam_list = list(self.beam_only_colours[self.adj_mat_to_key(graphs_np[k, -3:, -3:])] for k in range(graphs_np.shape[0]))
        
        undefined_dict = {'colour': 'k', 
                            'label': 'Undefined',
                            'plotted': False}
        
        beam_list = []
        for k in range(graphs_np.shape[0]):
            try:
                beam_list.append(self.beam_only_colours[self.adj_mat_to_key(graphs_np[k, -3:, -3:])])
            except KeyError:
                beam_list.append(undefined_dict)
            
        left_list = []
        for k in range(graphs_np.shape[0]):
            try:
                left_list.append(self.hand_connections[self.adj_mat_to_key(graphs_np[k, 0, :])])
            except KeyError:
                left_list.append(undefined_dict)
                
        right_list = []
        for k in range(graphs_np.shape[0]):
            try:
                right_list.append(self.hand_connections[self.adj_mat_to_key(graphs_np[k, 1, :])])
            except:
                right_list.append(undefined_dict)
        
        contact_dict = copy.deepcopy(undefined_dict)
        contact_dict['colour'] = 'b'
        contact_dict['label'] = "In contact"
        
        free_dict = copy.deepcopy(undefined_dict)
        free_dict['colour'] = 'r'
        free_dict['label'] = "Free space"
        
        left_free_space = []
        right_free_space = []
        for k in range(free_space_np.shape[0]):
            if free_space_np[k, 0] > 0.5:
                left_free_space.append(contact_dict)
            else:
                left_free_space.append(free_dict)
                
            if free_space_np[k, 1] > 0.5:
                right_free_space.append(contact_dict)
            else:
                right_free_space.append(free_dict)
                

        colour_list = [beam_list, left_list, right_list, left_free_space, right_free_space]
        
        fig, axes = plt.subplots(1, len(colour_list), figsize=(15, 4))
        fig.suptitle(title)
        for axis_counter, axis in enumerate(axes):
            axis.axis('equal')
            for k in range(x_np.shape[0]):
                if not colour_list[axis_counter][k]["plotted"]:
                    axis.scatter(x_np[k], y_np[k], color=colour_list[axis_counter][k]["colour"], label=colour_list[axis_counter][k]["label"])
                    colour_list[axis_counter][k]["plotted"] = True
                else:
                    axis.scatter(x_np[k], y_np[k], color=colour_list[axis_counter][k]["colour"])
            axis.legend()
        
        for c_list in colour_list:
            self.reset_colour_dict(c_list)
        
        return fig, axes

    def plot_figures(self, x_np, y_np, colour_list, title=''):
        fig, axes = plt.subplots(1, len(colour_list), figsize=(15, 4))
        fig.suptitle(title)
        for axis_counter, axis in enumerate(axes):
            axis.axis('equal')
            for k in range(x_np.shape[0]):
                if not colour_list[axis_counter][k]["plotted"]:
                    axis.scatter(x_np[k], y_np[k], color=colour_list[axis_counter][k]["colour"], label=colour_list[axis_counter][k]["label"])
                    colour_list[axis_counter][k]["plotted"] = True
                else:
                    axis.scatter(x_np[k], y_np[k], color=colour_list[axis_counter][k]["colour"])
            axis.legend()
        
        for c_list in colour_list:
            self.reset_colour_dict(c_list)
        return fig, axes
        