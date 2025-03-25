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
                BeamRobotEncoder,
                BeamRobotDecoder,
                BeamGraphClassifier).to(vae_params.device)
    
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
    
    if vae_params.split_encoder:
        robot_latents, beam_latents = latent_inspector.split_latent_dims(latent_dims)
        
        beam_only_latents = list(index - vae_params.beam_latent_dim for index in beam_latents)
        
        print("Robot latents: ", robot_latents)
        print("Beam latents:  ", beam_latents)
        
        # Apply sin and cos limit cycle to the robot latent space and then to the beam one
        dt = 0.1
        duration = 5.0
        steps = int(duration / 0.1)
        
        t = torch.linspace(0, duration, steps)
        
        latents = BeamRobotLatents()
        latents.robot.z = torch.zeros([steps, vae_params.robot_latent_dim], dtype=torch.float32).to(vae_params.device)
        latents.beams.z = torch.zeros([steps, vae_params.beam_latent_dim], dtype=torch.float32).to(vae_params.device)
        
        i, j = 0, 1
        Amp = 2.0 # Amplitude
        latents.robot.z[:, robot_latents[i]] = Amp * torch.cos(t)
        latents.robot.z[:, robot_latents[j]] = Amp * torch.sin(t)
        
        latents.beams.z[:, beam_only_latents[i]] = Amp * torch.cos(t)
        latents.beams.z[:, beam_only_latents[j]] = Amp * -torch.sin(t)
        
        x_out = model.decoder(latents, None)
        out_graph = model._classifier.forward(latents.z)    
            
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
        
        
        
    return 0 


if __name__ == "__main__":
    main()    
    