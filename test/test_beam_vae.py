import torch

from beam_data_gen.models.beam_robot_encoder import (BeamRobotEncoder, BeamRobotInputs)
from beam_data_gen.models.beam_robot_containers import (BeamRobotInputs, BeamRobotLatents, BeamRobotOutputs)
from beam_data_gen.models.beam_vae_pp import (BeamVaeParams, TrainParams,
                                              BeamVae, BeamDecoder, 
                                              BeamGraphClassifier)

def test_vae():
    
    beam_params = BeamVaeParams()
    train_params = TrainParams()   
       
    # Create model
    model = BeamVae(beam_params, 
                    train_params,
                    BeamRobotInputs,
                    BeamDecoder,
                    BeamGraphClassifier)
    
    # Model inputs
    batch_size = 2
    inputs = BeamRobotInputs()
    inputs.x_in = torch.zeros([batch_size, beam_params.input_dim], device=beam_params.device)
    inputs.x_out = torch.ones([batch_size, beam_params.output_dim], device=beam_params.device)
    inputs.graph_edge_targets = torch.ones([batch_size, beam_params.no_classifier_nodes, beam_params.no_classifier_nodes], device=beam_params.device)
    
    # Model forward pass
    latents, outputs = model.forward(inputs)
    
    # Test loss function
    loss = model.loss_func(inputs, latents, outputs)
    
    return 0


if __name__ == "__main__":
    test_vae()
