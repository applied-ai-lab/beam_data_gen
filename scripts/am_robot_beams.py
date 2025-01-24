import argparse
import time

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
from beam_data_gen.models.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.models.beam_vae_params import BeamVaeParams
from beam_data_gen.models.beam_train_params import TrainParams
from beam_data_gen.models.linear_classifier import LinearClassifier
from beam_data_gen.models.beam_robot_vae import (BeamVaeParams,
                                                BeamRobotVae, BeamRobotEncoder, BeamRobotLatents,
                                                BeamRobotInputs, BeamRobotOutputs,
                                                BeamDecoder, BeamGraphClassifier)
from beam_data_gen.latent_space.robot_beam_inspector import RobotBeamLSInspector


def main():
        
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)
    
    model = BeamRobotVae(vae_params, 
                train_params,
                BeamRobotEncoder,
                BeamDecoder,
                LinearClassifier).to(vae_params.device)
    
    model.load_state_dict(torch.load(vae_params.in_path))
    print(vae_params.in_path)
    
    
    # Inspect Latent Space
    latent_inspector = RobotBeamLSInspector(model, vae_params)
    
    # Load test data
    process_data = ProcessData(vae_params.pos_lims)
    poses, flat_adj = process_data(train_params.data_path, vae_params.graph_nodes)
    
    no_inputs = 1000
    rand_indices = np.random.choice(poses.shape[0], size=no_inputs)
    
    # Model inputs
    model_inputs = BeamRobotInputs()
    model_inputs.x_in = torch.tensor(poses[rand_indices, :], dtype=torch.float32).to(vae_params.device)
    model_inputs.graph_edge_targets = torch.tensor(flat_adj[rand_indices, :].reshape(-1, 
                                                                    vae_params.no_classifier_nodes, 
                                                                    vae_params.no_classifier_nodes), 
                                                    dtype=torch.float32).to(vae_params.device) 

    robot_beam_latent_dims = latent_inspector.find_latent_dims(model_inputs)
    
    # Sample from 2d circle
    no_samps = 500
    x, y = latent_inspector.sample_latent_values_from_unit_circle_2d(radius=2.5, no_samps=no_samps)
    
    latents_for_plotting: BeamRobotLatents = latent_inspector.sample_latent_space(2.5, 
                                                                                  robot_beam_latent_dims,
                                                                                  no_samps)
    # # Classify the graphs      
    graphs_for_plotting = torch.sigmoid(model.classifier(latents_for_plotting.z)).round()
    latent_inspector.plot_latents(x, y, graphs_for_plotting[:, 2:, 2:])
    plt.show()
            
    import pdb; pdb.set_trace()


    return 0


if __name__ == "__main__":
    main()
