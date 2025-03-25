import argparse
import time
import copy
import os

import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from torch.autograd import grad
from torch.nn import functional as F

from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader
from vae_planner.models.encoder_base import EncoderBase

from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected, RampGraph)
from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot)
from beam_data_gen.models.datasets.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams
from beam_data_gen.models.parameters.beam_train_params import TrainParams
from beam_data_gen.models.encoders.beam_robot_encoder import BeamRobotEncoder
from beam_data_gen.models.decoders.beam_robot_decoder import BeamRobotDecoder
from beam_data_gen.models.classifiers.space_classifier import SpaceClassifier
from beam_data_gen.models.vaes.beam_vae_pp import (BeamVaeParams,
                                              BeamVae, BeamEncoder, LatentVarsBase,
                                              BeamVaeInputs, BeamVaeOutputs,
                                              BeamDecoder, BeamGraphClassifier)
from beam_data_gen.latent_space.latent_inspector import BeamLSInspector


def main():
        
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)
    
    data_processor = ProcessData(np.array(vae_params.pos_lims))
    

    model = BeamVae(vae_params, 
                train_params,
                EncoderBase,
                BeamDecoder,
                SpaceClassifier).to(vae_params.device)
    
    model.load_state_dict(torch.load(vae_params.in_path))
    print(vae_params.in_path)
    
    
    # Inspect Latent Space
    latent_inspector = BeamLSInspector(model, vae_params)
    
    # Load test data
    process_data = ProcessData(vae_params.pos_lims)
    poses, flat_adj = process_data(train_params.data_path, vae_params.graph_nodes)
    
    no_inputs = 1000
    rand_indices = np.random.choice(poses.shape[0], size=no_inputs)

    
    # Model inputs
    model_inputs = BeamVaeInputs()
    model_inputs.x_in = torch.tensor(poses[rand_indices, :], dtype=torch.float32).to(vae_params.device)
    model_inputs.graph_edge_targets = torch.tensor(flat_adj[rand_indices, :].reshape(-1, 
                                                                    vae_params.no_classifier_nodes, 
                                                                    vae_params.no_classifier_nodes), 
                                                    dtype=torch.float32).to(vae_params.device) 

    (latent_dims, mean_var) = latent_inspector.find_latent_dims(model_inputs)
    
    # Sample from 2d circle
    no_samps = 300
    x, y = latent_inspector.sample_latent_values_from_unit_circle_2d(radius=2.5, no_samps=no_samps)
    
    latents_for_plotting = LatentVarsBase()
    latents_for_plotting.z = torch.zeros([no_samps, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    print(latent_dims)
    print(mean_var)
    
    for i in range(latent_dims.shape[0]):
        j = copy.deepcopy(i)
        while j < latent_dims.shape[0]:
            latents_for_plotting.z *= 0.0
            latents_for_plotting.z[:, i] = x
            latents_for_plotting.z[:, j] = y

            # Classify the graphs   
            graphs_for_plotting = torch.sigmoid(model._classifier.graph_forward(
                                        latents_for_plotting.z[:, 0:vae_params.robot_latent_dim])).round()
            
            free_space = torch.sigmoid(model._classifier.space_forward(
                                        latents_for_plotting.z[:, vae_params.robot_latent_dim:])).round()
            # Plot stuff
            title = f"Latent dim {i} and {j}, VAE {os.path.basename(vae_params.in_path)}"
            fig, axes = latent_inspector.plot_freespace_latents(x, y, graphs_for_plotting[:, :, :], free_space, title)
            
            file_path = os.path.join('figures', 'latent_space', f'VAE_11_latent_{i}_{j}')
            print(file_path)
            plt.savefig(file_path)
            
            plt.close(fig)
            
            j += 1
    
    return 0


if __name__ == "__main__":
    main()
