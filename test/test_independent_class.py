import torch

from beam_data_gen.models.classifiers.independent_classifier import IndependentClassifier
from beam_data_gen.models.parameters.beam_train_params import TrainParams
from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams


def main():
    
    beam_params = BeamVaeParams()
    train_params = TrainParams()   
    
    # Set vae latent dim
    beam_params.robot_latent_dim = 3 * 5
    beam_params.beam_latent_dim = 3 * 5

    # Create model
    classifier = IndependentClassifier(beam_params).to(beam_params.device)
    
    no_predictions = classifier.no_predictions
    
    batch_size = 2
    
    input = torch.zeros([batch_size, beam_params.latent_dim], dtype=torch.float32).to(beam_params.device)
    
    out_graph = classifier(input)
    
    import pdb; pdb.set_trace()
    
    return 0

if __name__ == "__main__":
    main()
