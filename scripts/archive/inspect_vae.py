import argparse

import torch
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader
from vae_planner.models.encoder_base import EncoderBase

from beam_data_gen.models.datasets.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.models.beam_vae_params import BeamVaeParams
from beam_data_gen.models.beam_train_params import TrainParams
from beam_data_gen.models.beam_vae_pp import (BeamVaeParams,
                                              BeamVae, BeamEncoder, LatentVarsBase,
                                              BeamVaeInputs, BeamVaeOutputs,
                                              BeamDecoder, BeamGraphClassifier)
from beam_data_gen.models.beam_robot_vae import(BeamRobotVae, BeamRobotLatents, BeamRobotEncoder)


def main():
        
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)
    
    data_processor = ProcessData(np.array(vae_args.pos_lims))
    
    with torch.no_grad():
        model = BeamRobotVae(vae_params, 
                    train_params,
                    BeamRobotEncoder,
                    BeamDecoder,
                    BeamGraphClassifier).to(vae_params.device)
        
        model.load_state_dict(torch.load(vae_params.in_path))
        
        latents = BeamRobotLatents()
        
        m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
        d = mujoco.MjData(m)
        
        with mujoco.viewer.launch_passive(m, d) as viewer:
            
            # Start loop and sample pose
            while viewer.is_running():
            
                latents.robot.z = torch.randn([1, vae_params.robot_latent_dim], dtype=torch.float32).to(vae_params.device)
                latents.beams.z = torch.randn([1, vae_params.beam_latent_dim], dtype=torch.float32).to(vae_params.device)
        
                out_pred = model.decoder(latents, None)
                out_graph = model.classifier(latents.z)
                
                print(torch.sigmoid(out_graph).round())
                
                denorm_out = data_processor.denorm_output(out_pred.x_pred)
                beams_out = denorm_out[:, 2*5:]
                robot_out = denorm_out[:, 0:2*5]
                
                # Set position
                d.qpos[0:3] = beams_out[0, 0:3].cpu().numpy()
                d.qpos[7:10] = beams_out[0, 5:8].cpu().numpy()
                d.qpos[14:17] = beams_out[0, 10:13].cpu().numpy()
                
                d.qpos[21:24] = robot_out[0, 0:3].cpu().numpy()
                d.qpos[28:31] = robot_out[0, 5:8].cpu().numpy()
                
                # Set orientation
                l1_z = R.from_euler("xyz", [0, 0, beams_out[0, 3].cpu().numpy()])
                l2_z = R.from_euler("xyz", [0, 0, beams_out[0, 8].cpu().numpy()])
                pa_z = R.from_euler("xyz", [0, 0, beams_out[0, 13].cpu().numpy()])
                
                robot_left = R.from_euler("xyz", [0, 0, robot_out[0, 3].cpu().numpy()])
                robot_right = R.from_euler("xyz", [0, 0, robot_out[0, 8].cpu().numpy()])
                
                d.qpos[4:7]   = l1_z.as_quat()[0:3]
                d.qpos[3]   = l1_z.as_quat()[3]
                d.qpos[11:14] = l2_z.as_quat()[0:3]
                d.qpos[10] = l2_z.as_quat()[3]
                d.qpos[18:21] = pa_z.as_quat()[0:3]
                d.qpos[17] = pa_z.as_quat()[3]    
                
                d.qpos[25:28] = robot_left.as_quat()[0:3]
                d.qpos[24] = robot_left.as_quat()[3]
                        
                d.qpos[32:35] = robot_right.as_quat()[0:3]
                d.qpos[31] = robot_right.as_quat()[3]
                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
        
                import pdb; pdb.set_trace()   


    return 0


if __name__ == "__main__":
    main()
