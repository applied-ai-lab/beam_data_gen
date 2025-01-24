import os
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from vae_planner.argparse_yaml_loader.yaml_loader import YamlLoader
from vae_planner.models.encoder_base import EncoderBase

from beam_data_gen.models.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.models.beam_vae_params import BeamVaeParams
from beam_data_gen.models.beam_train_params import TrainParams
from beam_data_gen.models.beam_vae_pp import BeamVae
from beam_data_gen.models.beam_robot_vae import (BeamVaeParams,
                                                BeamRobotVae, BeamRobotEncoder, BeamRobotLatents,
                                                BeamRobotInputs, BeamRobotOutputs,
                                                BeamDecoder, BeamGraphClassifier)


# Define training and testing functions
def train(model: BeamRobotVae, dataloader, optimizer):

    model.train()
    total_loss = 0.0
    kl_loss = 0.0
    mse_loss = 0.0
    graph_loss = 0.0
    for x_in, x_out, adj_mat in dataloader:
        optimizer.zero_grad()

        inputs = model.vae_inputs
        inputs.x_in = x_in
        inputs.x_out = x_out
        inputs.graph_edge_targets = adj_mat

        latents, outputs = model(inputs)

        loss = model.loss_func(inputs, latents, outputs)
        loss.tot_loss.backward()
        
        total_loss += loss.tot_loss.item()
        kl_loss += loss.kl.item()
        mse_loss += loss.mse.item()
        graph_loss += loss.cross_entropy.item()

        optimizer.step()
    return (total_loss / len(dataloader.dataset), 
            kl_loss / len(dataloader.dataset), 
            mse_loss / len(dataloader.dataset),
            graph_loss / len(dataloader.dataset))

# Define training and testing functions
def test(model: BeamRobotVae, dataloader):

    model.eval()
    total_loss = 0.0
    kl_loss = 0.0
    mse_loss = 0.0
    graph_loss = 0.0

    with torch.no_grad():
        for x_in, x_out, adj_mat in dataloader:
            inputs = model.vae_inputs
            inputs.x_in = x_in
            inputs.x_out = x_out
            inputs.graph_edge_targets = adj_mat

            latents, outputs = model(inputs)

            loss = model.loss_func(inputs, latents, outputs)
            
            total_loss += loss.tot_loss.item()
            kl_loss += loss.kl.item()
            mse_loss += loss.mse.item()
            graph_loss += loss.cross_entropy.item()

    return (total_loss / len(dataloader.dataset), 
            kl_loss / len(dataloader.dataset), 
            mse_loss / len(dataloader.dataset),
            graph_loss / len(dataloader.dataset)
            )


def main():
    # Config file for loading data
    parser = argparse.ArgumentParser(description='Process Data')
    vae_args, train_args = YamlLoader(parser).return_args()

    writer = SummaryWriter()

    vae_params = BeamVaeParams(vae_args)
    train_params = TrainParams(train_args)

    torch.manual_seed(train_params.seed)
    
    ##########################################
    # Process data
    process_data = ProcessData(np.array(vae_params.pos_lims))
    poses, flat_adj = process_data(train_params.data_path, ["robot_left_hand", "robot_right_hand", "l_beam_1", "l_beam_2", "l_pin_A"])

    # Create dataset and dataloaders
    dataset_class = BeamDataset(poses, flat_adj, device=vae_params.device)
    ##########################################
    
    # Split training and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_class, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=train_params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_params.batch_size, shuffle=True)

    model = BeamVae(vae_params, 
                        train_params,
                        EncoderBase,
                        BeamDecoder,
                        BeamGraphClassifier).to(vae_params.device)
    
    if train_params.read_from_file:
        model.load_state_dict(torch.load(vae_params.in_path))

    if not os.path.exists(vae_params.out_dir):
        os.makedirs(vae_params.out_dir)

    optimizer = optim.Adam(model.parameters(), lr=train_params.lr)

    # Train/test loop
    for epoch in range(1, train_params.epochs, 1):
        train_losses = train(model, train_dataloader, optimizer)
        test_losses = test(model, test_dataloader)

        print(f"Epoch {epoch}/{train_params.epochs}, Train Loss: {train_losses[0]:.4f}, Test Loss: {test_losses[0]:.4f}")
        if epoch % train_params.log_interval == 0:
            # Save model
            model_out_path = os.path.join(vae_params.out_dir, "vae_epoch_" + str(epoch) + ".pt")
            torch.save(model.state_dict(), model_out_path)

        writer.add_scalar('train_loss', train_losses[0], global_step=epoch)
        writer.add_scalar('train_kl',   train_losses[1], global_step=epoch)
        writer.add_scalar('train_mse',  train_losses[2], global_step=epoch)
        writer.add_scalar('train_graph', train_losses[3], global_step=epoch)

        writer.add_scalar('test_loss', test_losses[0], global_step=epoch)
        writer.add_scalar('test_kl',   test_losses[1], global_step=epoch)
        writer.add_scalar('test_mse',  test_losses[2], global_step=epoch)
        writer.add_scalar('test_graph', test_losses[3], global_step=epoch)

    return 0


if __name__ == "__main__":
    main()
