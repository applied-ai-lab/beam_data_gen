import argparse

import torch

from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader

from beam_data_gen.models.parameters.beam_vae_params import BeamVaeParams
from beam_data_gen.models.parameters.beam_train_params import TrainParams
from beam_data_gen.models.containers.beam_containers import LatentVarsBase
from beam_data_gen.models.classifiers.space_classifier import SpaceClassifier


def main():
    
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)
    
    classifier = SpaceClassifier(vae_params).to(vae_params.device)
    
    latents = LatentVarsBase()
    latents.z = torch.zeros([1, vae_params.latent_dim], dtype=torch.float32).to(vae_params.device)
    
    logits = classifier(latents.z)
    
    print(logits)
    
    import pdb
    pdb.set_trace()    
    
    return 0


if __name__ == "__main__":
    main()
