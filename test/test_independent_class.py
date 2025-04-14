import torch
import torch.nn as nn
import torch.optim as optim 

from beam_data_gen.models.classifiers.independent_classifier import IndependentClassifier
from beam_data_gen.models.parameters.beam_train_params import TrainParams
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


def print_params(model):
    for name, param in model.named_parameters():
            print(name, param.shape)
            print(param.data.clone())  # or param.data for raw values
    return


def main():
    
    beam_params = BeamVaeParams()
    train_params = TrainParams()   
    
    # Set vae latent dim
    beam_params.robot_latent_dim = 3 * 5
    beam_params.beam_latent_dim = 3 * 5

    # Create model
    classifier = IndependentClassifier(beam_params).to(beam_params.device)
    
    optimizer = optim.Adam(classifier.parameters(), lr=train_params.lr)
    
    batch_size = 2
    
    input = torch.ones([batch_size, beam_params.latent_dim], dtype=torch.float32).to(beam_params.device)
    
    target = torch.ones([batch_size, 3, 3], dtype=torch.float32).to(beam_params.device)
    
    loss = nn.MSELoss()
    
    print_params(classifier)

    for _ in range(1):
        out_graph = classifier(input)
        
        loss_value = loss(target, out_graph)
        loss_value.backward()    
        
        optimizer.step()
        
        print_params(classifier)
        
    import pdb; pdb.set_trace()
    
    return 0

if __name__ == "__main__":
    main()
