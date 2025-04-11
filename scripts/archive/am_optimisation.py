import copy
import argparse
import time

import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import mujoco
import mujoco.viewer

from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader
from beam_data_gen.activation_maximisation.beam_act_max import (BeamActMax, BeamSetPoint, 
                                                                ActMaxParams, ActMaxOutput)
from beam_data_gen.models.vaes.beam_vae_pp import (BeamVae, BeamVaeParams, TrainParams, 
                                            BeamDecoder, BeamGraphClassifier)
from vae_planner.models.encoder_base import EncoderBase
from vae_planner.models.container_base import LatentVarsBase
from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot)
from beam_data_gen.models.datasets.beam_dataset import ProcessData
from beam_data_gen.simulator.beam_robot_sim import BeamRobotSim


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
    
    # Activation maximisation
    act_max_params = ActMaxParams(nn.BCEWithLogitsLoss(), 1e-2, 1000, 0.2)
    beam_actmax = BeamActMax(act_max_params, torch.device("cuda"))
    
    z0 = -1.0 * torch.ones([vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    l_pin_removed_robot.add_hand("robot_left_hand", "l_beam_1")
    l_pin_removed_robot.add_hand("robot_right_hand", "l_beam_2")
    
    l_connected_robot.add_hand("robot_left_hand", "l_beam_2")
    l_connected_robot.add_hand("robot_right_hand", "l_beam_1")
    
    set_points = [BeamSetPoint(l_pin_removed_robot.A, 100), 
                BeamSetPoint(l_connected_robot.A, 50)]
    
    out = beam_actmax.optimise(model, z0, set_points)
    
    latent_traj = LatentVarsBase()
    latent_traj.z = out.z
    x_traj = model.decoder(latent_traj, None)
    
    traj_len = x_traj.x_pred.shape[0]
    
    counter = 0
    
    # Visualise with mujoco
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        input("continue")        
        # Start loop and sample pose
        while viewer.is_running() and counter < traj_len:
            # Decoder the prediction
            beam_sim.decode_x(d, x_traj.x_pred[counter:counter+1, :])
                            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)    

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()   
            
            # Rudimentary time keeping, will drift relative to wall clock.
            time.sleep(0.1)
            
            counter += 1
    
    
    plt.figure()
    plt.plot(out.loss.detach().cpu().numpy())
    
    plt.figure()
    plt.plot(out.z.detach().cpu().numpy())
    plt.show()
    
    import pdb
    pdb.set_trace()
    
    return 0


if __name__ == "__main__":
    main()
