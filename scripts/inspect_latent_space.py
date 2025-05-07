import argparse
import time
import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
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
from beam_data_gen.models.datasets.trajectory_dataset import TrajectoryDataset, ProcessTrajectories
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams
from beam_data_gen.models.parameters.beam_train_params import TrainParams
from beam_data_gen.models.classifiers.independent_classifier import IndependentClassifier
from beam_data_gen.models.encoders.beam_robot_encoder import BeamRobotEncoder
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

    model = BeamVae(vae_params, 
                train_params,
                EncoderBase,
                BeamDecoder,
                IndependentClassifier).to(vae_params.device)
    
    model.load_state_dict(torch.load(vae_params.in_path))
    print(vae_params.in_path)
    
    
    # Inspect Latent Space
    latent_inspector = BeamLSInspector(model, vae_params)
    
    # Process data    
    process_data = ProcessTrajectories(np.array(vae_params.pos_lims), device=vae_params.device)
    poses, flat_adj = process_data(train_params.data_path, vae_params.graph_nodes)
    
    no_inputs = 1000
    dataset = TrajectoryDataset(poses, flat_adj, vae_params.no_inputs, vae_params.no_outputs)
    loader = DataLoader(dataset, batch_size=no_inputs, shuffle=True)
    
    x_in, x_out, adj_mat = next(iter(loader))
    
    # Model inputs
    model_inputs = BeamVaeInputs()
    model_inputs.x_in = x_in
    model_inputs.graph_edge_targets = adj_mat

    (latent_dims, mean_var) = latent_inspector.find_latent_dims(model_inputs)
    
    # Sort latents
    binned_latents = latent_inspector.bin_latent_dims(latent_dims, model._classifier._independent_latent_dim)
    
    # Sample from 2d circle
    no_samps = 300
    x, y = latent_inspector.sample_latent_values_from_unit_circle_2d(radius=2.5, no_samps=no_samps)
    
    latents_for_plotting = LatentVarsBase()
    latents_for_plotting.z = torch.zeros([no_samps, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    print(latent_dims)
    print(mean_var)
    
    for sub_latent, small_latents in binned_latents.items():
    
        for i in range(len(small_latents)):
            j = copy.deepcopy(i)
            while j < len(small_latents):
                latents_for_plotting.z[:, small_latents[i]] = x
                latents_for_plotting.z[:, small_latents[j]] = y

                # Classify the graphs      
                graphs_for_plotting = torch.sigmoid(model.classifier(latents_for_plotting.z)).round()
                title = f"Latent dim {i} and {j}, VAE {os.path.basename(vae_params.in_path)}"
                fig, axes = latent_inspector.plot_latents(x, y, graphs_for_plotting[:, :, :], title)
                
                file_dir = os.path.join('figures', 'latent_space', f'latent_{sub_latent}')
                
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                
                file_path = os.path.join(file_dir,  f'VAE_latent_{small_latents[i]}_{small_latents[j]}')
                print(file_path)
                plt.savefig(file_path)
                
                plt.close(fig)
                
                j += 1
    
    return 0


if __name__ == "__main__":
    main()