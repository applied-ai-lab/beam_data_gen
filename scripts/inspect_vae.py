import argparse

import torch

from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader
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
    
    with torch.no_grad():
        model = BeamVae(vae_params, 
                    train_params,
                    BeamEncoder,
                    BeamDecoder,
                    BeamGraphClassifier).to(vae_params.device)
        
        model.load_state_dict(torch.load(vae_params.in_path))
        
        latents = LatentVarsBase()
        latents.z = torch.zeros([1000, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
        
        out_pred = model.decoder(latents, None)
        out_graph = model.classifier(latents.z)
        
        print(torch.sigmoid(out_graph).round())
        
        import pdb; pdb.set_trace()   


    return 0


if __name__ == "__main__":
    main()
