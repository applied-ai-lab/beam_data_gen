import numpy as np
import torch

from beam_data_gen.models.datasets.beam_dataset import BeamDataset, ProcessData


def main():
    
    path = "data/robot_graphs/"
    
    process_data = ProcessData(np.array([1.0, 1.0, 0.25]))
    poses, adj = process_data(path, ["robot_left_hand", "robot_right_hand", "l_beam_1", "l_beam_2", "l_pin_A"])
    
    # Data loader
    dataset = BeamDataset(poses, adj, torch.device('cuda'))
    
    x_in, x_out, adj_mat = dataset.__getitem__(1)
    
    import pdb
    pdb.set_trace()
    
    return 0


if __name__ == "__main__":
    main()
    
