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
from vae_planner.activation_maximisation.act_max import (ActivationMaximisation, ActMaxParams, ActMaxOutput)

from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected, RampGraph)
from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot)
from beam_data_gen.models.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.models.beam_vae_params import BeamVaeParams
from beam_data_gen.models.beam_train_params import TrainParams
from beam_data_gen.models.beam_vae_pp import (BeamVaeParams,
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
                BeamGraphClassifier).to(vae_params.device)
    
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
    
    # i, j = 0, 1
    # latents_for_plotting.z[:, latent_dims[0]] = x
    # latents_for_plotting.z[:, latent_dims[1]] = y

    # # Classify the graphs      
    # graphs_for_plotting = torch.sigmoid(model.classifier(latents_for_plotting.z)).round()
    # title = f"Latent dim {i} and {j}, VAE {os.path.basename(vae_params.in_path)}"
    # fig, axes = latent_inspector.plot_latents(x, y, graphs_for_plotting[:, :, :], title)
    
    # file_path = os.path.join('figures', 'latent_space', f'VAE_11_latent_{i}_{j}')
    # plt.show()
            
        
    # AM for ls
    act_max_params = ActMaxParams(nn.BCEWithLogitsLoss(), 1e-2, 100, 0.2)
    act_max = ActivationMaximisation(act_max_params, vae_params.device)    
    
    latents = LatentVarsBase()
    
    m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
    d = mujoco.MjData(m)
    
    latents.z = -1.0 * torch.ones([1, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device).requires_grad_()
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
    graph_target = torch.tensor([adj_mat], dtype=torch.float32)
    graph_target = graph_target.to(vae_params.device)
    
    print(adj_mat)
    print(torch.sigmoid(out_graph).round())
    
    loss = torch.tensor([1000.0]).to(vae_params.device)
        
    # Visualisation runs
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        input("continue")
        
        step_start = time.time()
        
        # Start loop and sample pose
        while viewer.is_running():
            while torch.norm(loss) > act_max_params.stop_criterion:
                
                latents.z, grad_features, loss = act_max.advance(model._classifier.forward, 
                                                                latents.z,
                                                                graph_target)
                
                # Loss                
                out_graph = model._classifier.forward(latents.z)
                
                print(torch.sigmoid(out_graph).round())
                
                # Graph features
                grad_list.append(grad_features.detach().cpu().numpy().squeeze())
                latent_list.append(latents.z.detach().cpu().numpy().squeeze())
                
                out_pred = model.decoder(latents, None)
                out_graph = model.classifier(latents.z)
                
                denorm_out = data_processor.denorm_output(out_pred.x_pred)[:, 10:]
                
                # Set position
                d.qpos[0:3] = denorm_out[0, 0:3].cpu().detach().numpy()
                d.qpos[7:10] = denorm_out[0, 5:8].cpu().detach().numpy()
                d.qpos[14:17] = denorm_out[0, 10:13].cpu().detach().numpy()
                
                # Set orientation
                l1_z = R.from_euler("xyz", [0, 0, denorm_out[0, 3].cpu().detach().numpy()])
                l2_z = R.from_euler("xyz", [0, 0, denorm_out[0, 8].cpu().detach().numpy()])
                pa_z = R.from_euler("xyz", [0, 0, denorm_out[0, 13].cpu().detach().numpy()])
                
                d.qpos[4:7]   = l1_z.as_quat()[0:3]
                d.qpos[3]   = l1_z.as_quat()[3]
                d.qpos[11:14] = l2_z.as_quat()[0:3]
                d.qpos[10] = l2_z.as_quat()[3]
                d.qpos[18:21] = pa_z.as_quat()[0:3]
                d.qpos[17] = pa_z.as_quat()[3]            
                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()   
                
                # Rudimentary time keeping, will drift relative to wall clock.
                time.sleep(0.1)
            
            # Stack gradients
            gradient_traj = np.vstack(grad_list)
            latent_traj = np.vstack(latent_list)
            
            import pdb; pdb.set_trace()


    return 0


if __name__ == "__main__":
    main()
