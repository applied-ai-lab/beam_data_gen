import argparse
import time

import torch
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
    poses, flat_adj = process_data(train_params.data_path, ["l_beam_1", "l_beam_2", "l_pin_A"])
    
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
    no_samps = 1000
    x, y = latent_inspector.sample_latent_values_from_unit_circle_2d(radius=2.5, no_samps=no_samps)
    
    latents_for_plotting = LatentVarsBase()
    latents_for_plotting.z = torch.zeros([no_samps, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    print(latent_dims)
    
    latents_for_plotting.z[:, latent_dims[0]] = x
    latents_for_plotting.z[:, latent_dims[1]] = y

    # Classify the graphs      
    graphs_for_plotting = torch.sigmoid(model.classifier(latents_for_plotting.z)).round()
    latent_inspector.plot_latents(x, y, graphs_for_plotting)
        
    # AM for ls
    latents = LatentVarsBase()
    
    m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
    d = mujoco.MjData(m)
    
    latents.z = 2.0 * torch.ones([1, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device).requires_grad_()
    latent_list = []
    
    grad_features = 100 * torch.ones(latents.z.shape, dtype=torch.float32).to(vae_params.device)
    grad_list = []
    
    out_pred = model.decoder(latents, None)
    out_graph = model.classifier(latents.z)
            
    denorm_out = data_processor.denorm_output(out_pred.x_pred)
    
    # Graph Target
    graph_target = torch.tensor([l_connected_graph.A], dtype=torch.float32)
    graph_target = graph_target.to(vae_params.device)
    
    termination_criteria = 0.05     
    
    # Visualisation runs
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        input("continue")
        
        step_start = time.time()
        
        # Start loop and sample pose
        while viewer.is_running():
            while torch.norm(grad_features) > termination_criteria:
                # Loss
                out_graph = model._classifier.forward(latents.z)
                loss = F.binary_cross_entropy_with_logits(out_graph, graph_target)
                
                print(torch.sigmoid(out_graph).round())
                
                # Graph features
                grad_features = grad(outputs=loss, inputs=latents.z, retain_graph=True)[0]
                grad_list.append(grad_features.detach().cpu().numpy().squeeze())
                    
                latents.z = latents.z - 1.0e-3 * grad_features
                latent_list.append(latents.z.detach().cpu().numpy().squeeze())
                
                out_pred = model.decoder(latents, None)
                out_graph = model.classifier(latents.z)
                
                denorm_out = data_processor.denorm_output(out_pred.x_pred)
                
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
