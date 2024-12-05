import argparse
import time

import torch
import numpy as np
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

def main():
        
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)
    
    data_processor = ProcessData(np.array([0.6, 0.6, 0.08]))
    

    model = BeamVae(vae_params, 
                train_params,
                EncoderBase,
                BeamDecoder,
                BeamGraphClassifier).to(vae_params.device)
    
    model.load_state_dict(torch.load(vae_params.in_path))
    
    latents = LatentVarsBase()
    
    m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
    d = mujoco.MjData(m)
    
    latents.z = torch.ones([1, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device).requires_grad_()
    
    out_pred = model.decoder(latents, None)
    out_graph = model.classifier(latents.z)
            
    denorm_out = data_processor.denorm_output(out_pred.x_pred)
    
    # Graph Target
    graph_target = torch.tensor([l_pin_removed.A], dtype=torch.float32)
    graph_target = graph_target.to(vae_params.device)
        
    
    # Visualisation runs
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        import pdb; pdb.set_trace()
        
        step_start = time.time()
        
        # Start loop and sample pose
        while viewer.is_running():
            
            # Loss
            out_graph = model._classifier.forward(latents.z)
            loss = F.binary_cross_entropy_with_logits(out_graph, graph_target)
            
            print(torch.sigmoid(out_graph).round())
            
            # Graph features
            grad_features = grad(outputs=loss, inputs=latents.z, retain_graph=True)[0]
            print(grad_features)
            
                   
            latents.z = latents.z - 1.0e-2 * grad_features
            
            out_pred = model.decoder(latents, None)
            out_graph = model.classifier(latents.z)
            
            denorm_out = data_processor.denorm_output(out_pred.x_pred)
            
            # Set position
            d.qpos[0:3] = denorm_out[0, 0:3].cpu().detach().numpy()
            d.qpos[7:10] = denorm_out[0, 5:8].cpu().detach().numpy()
            d.qpos[14:17] = denorm_out[0, 9:12].cpu().detach().numpy()
            
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


    return 0


if __name__ == "__main__":
    main()
