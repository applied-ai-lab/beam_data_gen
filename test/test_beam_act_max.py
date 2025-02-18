import copy
import argparse

import torch
from torch import nn
from matplotlib import pyplot as plt

from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader
from beam_data_gen.activation_maximisation.beam_act_max import (BeamActMax, BeamSetPoint, 
                                                                ActMaxParams, ActMaxOutput)
from beam_data_gen.models.beam_vae_pp import (BeamVae, BeamVaeParams, TrainParams, 
                                            BeamDecoder, BeamGraphClassifier)
from vae_planner.models.encoder_base import EncoderBase
from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot)



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
                BeamGraphClassifier).to(vae_params.device)
    
    model.load_state_dict(torch.load(vae_params.in_path))
    print(vae_params.in_path)
    
    # Activation maximisation
    act_max_params = ActMaxParams(nn.BCEWithLogitsLoss(), 1e-2, 1000, 0.2)
    beam_actmax = BeamActMax(act_max_params, torch.device("cuda"))
    
    A_mat = beam_actmax.vel_constraint(10, 5)
    
    print(A_mat)   
    
    z0 = torch.zeros([vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    l_pin_removed_robot.add_hand("robot_left_hand", "l_beam_1")
    l_pin_removed_robot.add_hand("robot_right_hand", "l_beam_2")
    
    set_points = [BeamSetPoint(l_pin_removed_robot.A, 100), 
                BeamSetPoint(l_disconnected_robot.A, 50)]
    
    out = beam_actmax.optimise(model, z0, set_points)
    
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
