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
    

    model = BeamVae(vae_params, 
                train_params,
                EncoderBase,
                BeamDecoder,
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
    
        
    # AM for ls
    act_max_params = ActMaxParams(nn.BCEWithLogitsLoss(), 1.0e-1, 100, 0.2)
    act_max = BeamActMax(act_max_params, vae_params.device)    
    
    latents = LatentVarsBase()
    
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)
    
    latents.z = 1.0 * torch.ones([1, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device).requires_grad_(True)
    latent_list = []
    
    grad_features = 100 * torch.ones(latents.z.shape, dtype=torch.float32).to(vae_params.device)
    grad_list = []
    
    out_pred = model.decoder(latents, None)
    out_graph = model.classifier(latents.z)
            
    denorm_out = data_processor.denorm_output(out_pred.x_pred)
    
    # Graph Target
    graph = l_pin_removed_robot
    graph.add_hand("robot_left_hand", "l_beam_1")
    graph.add_hand("robot_right_hand", "l_beam_2")
    
    adj_mat = graph.A
    
    graph_target = torch.tensor([adj_mat], dtype=torch.float32).to(vae_params.device)
    
    print(adj_mat)
    print(torch.sigmoid(out_graph).round())
    
    loss = torch.tensor([1000.0]).to(vae_params.device)
    loss_lst = []    
    
    loss_graph_lst = []
    loss_mse_lst = []
    
    init_pin_pose = out_pred.x_pred[:, -5:].clone()   
    mse_loss =nn.MSELoss(reduction='sum')
        
    latent_torch_list = []
    latents.z = latents.z.clone().detach().requires_grad_(True)
    latent_list.append(latents.z.detach().cpu().numpy().squeeze())
    latent_torch_list.append(latents.z.clone())    
    
    # Create optimiser
    optimizer = optim.SGD([latents.z], lr=act_max_params.lr)
    
    counter = 0
    
    while torch.norm(loss) > act_max_params.stop_criterion and counter < act_max_params.max_iters:
        graph_hat = model._classifier.forward(latents.z)
        loss = act_max_params.loss_func(graph_hat, graph_target)
        optimizer.zero_grad()                
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Store some values
        loss_lst.append(loss.detach().cpu().numpy())
        latent_list.append(latents.z.detach().cpu().numpy().squeeze())
        latent_torch_list.append(latents.z.clone())
        
        counter += 1
        
    # Set new graph target
    # Graph Target
    graph = l_pin_removed_robot
    graph.add_hand("robot_left_hand", "l_beam_1")
    graph.add_hand("robot_right_hand", "l_pin_A")
    graph_target = torch.tensor([graph.A], dtype=torch.float32).to(vae_params.device)
    
    loss = torch.tensor([1000.0]).to(vae_params.device)    
    counter = 0
    
    while torch.norm(loss) > act_max_params.stop_criterion and counter < act_max_params.max_iters:
        graph_hat = model._classifier.forward(latents.z)
        loss = act_max_params.loss_func(graph_hat, graph_target)
        optimizer.zero_grad()                
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Store some values
        loss_lst.append(loss.detach().cpu().numpy())
        latent_list.append(latents.z.detach().cpu().numpy().squeeze())
        latent_torch_list.append(latents.z.clone())
        
        counter += 1
        
    
    # Set new graph target
    # Graph Target
    graph = l_connected_robot
    graph.add_hand("robot_left_hand", "l_beam_1")
    graph.add_hand("robot_right_hand", "l_pin_A")
    graph_target = torch.tensor([graph.A], dtype=torch.float32).to(vae_params.device)
    
    loss = torch.tensor([1000.0]).to(vae_params.device)    
    counter = 0
    
    while torch.norm(loss) > act_max_params.stop_criterion and counter < act_max_params.max_iters:
        graph_hat = model._classifier.forward(latents.z)
        loss = act_max_params.loss_func(graph_hat, graph_target)
        optimizer.zero_grad()                
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Store some values
        loss_lst.append(loss.detach().cpu().numpy())
        latent_list.append(latents.z.detach().cpu().numpy().squeeze())
        latent_torch_list.append(latents.z.clone())
        
        counter += 1
    
    latents_torch = torch.cat(latent_torch_list,dim=0)
    
    latents = LatentVarsBase()
    latents.z = latents_torch
    
    x_out = model.decoder(latents, None)
    out_graph = model._classifier.forward(latents.z)
    
    # Optimise z traj again for primal
    out  = act_max.optimise_primal(model, latents, out_graph.detach().clone())
    
    latents.z = out.z
    
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
                time.sleep(0.1)
            
            
            # Stack gradients
            latent_traj = np.vstack(latent_list)
            loss_traj = np.vstack(loss_lst)
            
            latent_primal_traj = latents.z.detach().cpu().numpy()

            print(np.std(latent_traj, 0))
            
            i, j = latent_dims[0], latent_dims[1]
            # i, j = 3, 4
            latents_for_plotting.z[:, i] = x
            latents_for_plotting.z[:, j] = y

            # Classify the graphs      
            graphs_for_plotting = torch.sigmoid(model.classifier(latents_for_plotting.z)).round()
            title = f"Latent dim {i} and {j}, VAE {os.path.basename(vae_params.in_path)}"
            fig, axes = latent_inspector.plot_latents(x, y, graphs_for_plotting[:, :, :], title)
            
            prop_cycle = plt.rcParams['axes.prop_cycle']
            
            # Plot 2D trajectories
            for axis in axes:
                # axis.plot(latent_traj[:, i], latent_traj[:, j], marker="*")
                axis.plot(latent_primal_traj[:, i], latent_primal_traj[:, j], alpha=1.0, color="k")
                
            plt.figure()
            plt.plot(loss_traj, label="total")
            plt.plot(out.loss.detach().cpu().numpy())
            plt.legend()
            
            plt.figure()
            for k, sty in zip(range(len(latent_dims)), cycle(prop_cycle)):
                plt.plot(latent_traj[:, k], **sty, marker="*")
                plt.plot(latent_primal_traj[:, k], **sty)
            
            plt.show()        
            
                        
            import pdb; pdb.set_trace()


    return 0


if __name__ == "__main__":
    main()
