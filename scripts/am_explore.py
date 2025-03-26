import argparse
import time
import copy
import os
from itertools import cycle

import torch
from torch import nn
from torch import optim
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
from vae_planner.activation_maximisation.act_max import (ActMaxParams, ActMaxOutput)

from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot)
from beam_data_gen.models.datasets.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams
from beam_data_gen.models.parameters.beam_train_params import TrainParams
from beam_data_gen.models.encoders.beam_robot_encoder import BeamRobotEncoder
from beam_data_gen.models.decoders.beam_robot_decoder import BeamRobotDecoder
from beam_data_gen.models.containers.beam_robot_containers import BeamRobotLatents
from beam_data_gen.models.classifiers.space_classifier import SpaceClassifier
from beam_data_gen.models.vaes.beam_vae_pp import (BeamVaeParams,
                                              BeamVae, BeamEncoder, LatentVarsBase,
                                              BeamVaeInputs, BeamVaeOutputs,
                                              BeamDecoder, BeamGraphClassifier)
from beam_data_gen.latent_space.latent_inspector import BeamLSInspector
from beam_data_gen.simulator.beam_robot_sim import BeamRobotSim
from beam_data_gen.activation_maximisation.beam_act_max import BeamActMax


def main():
        
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)
    
    data_processor = ProcessData(np.array(vae_params.pos_lims))
    
    beam_sim = BeamRobotSim(data_processor)
    
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)
    
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
    dataset = BeamDataset(poses, flat_adj, vae_params.device)
    loader = DataLoader(dataset, batch_size=no_inputs, shuffle=True)
    
    x_in, x_out, adj_mat = next(iter(loader))
    
    # Model inputs
    model_inputs = BeamVaeInputs()
    model_inputs.x_in = x_in
    model_inputs.graph_edge_targets = adj_mat

    (latent_dims, mean_var) = latent_inspector.find_latent_dims(model_inputs)
    
    # Sample from 2d circle
    no_samps = 300
    x, y = latent_inspector.sample_latent_values_from_unit_circle_2d(radius=2.5, no_samps=no_samps)
    
    latents_for_plotting = LatentVarsBase()
    latents_for_plotting.z = torch.zeros([no_samps, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    print(latent_dims)
    print(mean_var)
    
    # Trajectory Parameters
    dt = 0.1
    duration = 5.0
    steps = int(duration / 0.1)
        
    t = torch.linspace(0, duration, steps)
    
    latents = LatentVarsBase()
    latents.z = torch.zeros([steps, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    # Latent dims to explore
    beam_ls = [0, 3] # 2D planning space
    
    # Hand planning spaces
    left_hand = [1, 7] # [Beam LS, Contact LS]
    right_hand = [5, 11] # [Beam LS, Contact LS]
    
    # Start fully assembled
    latents.z[:, beam_ls[0]] = -1.0 * torch.ones([steps], dtype=torch.float32).to(vae_params.device)
    latents.z[:, beam_ls[1]] = 1.0 * torch.ones([steps], dtype=torch.float32).to(vae_params.device)
    
    ## LEFT HAND
    # Lift left hand
    latents.z[0:steps // 2, left_hand[1]] = torch.linspace(0, -2, steps // 2, dtype=torch.float32).to(vae_params.device)
    latents.z[steps // 2:, left_hand[1]] = -2.0 * torch.ones([steps // 2], dtype=torch.float32).to(vae_params.device)
    
    # Move left hand to another item
    latents.z[steps // 2:, left_hand[0]] = torch.linspace(0, -2, steps // 2, dtype=torch.float32).to(vae_params.device)
    
    ## RIGHT HAND    
    # Move left hand to another item
    latents.z[:, beam_ls[0]] = torch.linspace(1.0, 2.0, steps, dtype=torch.float32).to(vae_params.device)
    
    
    # Decode outputs
    x_out = model.decoder(latents, None)
    out_graph = model._classifier.graph_forward(latents.z)    
    
    print("Graph Prediction:")
    print(torch.sigmoid(out_graph[0, :, :]).round().detach().cpu().numpy())
        
    # Visualisation runs
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        input("continue")
        
        # Start loop and sample pose
        while viewer.is_running():
            for k in range(latents.z.shape[0]):
                
                # Decoder the prediction
                beam_sim.decode_x(d, x_out.x_pred[k:k+1, :])
                                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()   
                
                # Rudimentary time keeping, will drift relative to wall clock.
                time.sleep(dt)
                
            break
    
    plt.figure()
    plt.plot(latents.z.detach().cpu().numpy())
    plt.show()
        
    return 0 


if __name__ == "__main__":
    main()    
    