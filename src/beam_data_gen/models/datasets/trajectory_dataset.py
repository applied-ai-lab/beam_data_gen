import os
import copy
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from beam_data_gen.models.datasets.process_trajectories import ProcessTrajectories


class TrajectoryDataset(Dataset):
    def __init__(self, 
                 pose_traj: Dict[int, torch.tensor], 
                 adj_traj: Dict[int, torch.tensor],
                 no_inputs: int,
                 no_outputs: int):
        super().__init__()
        
        self._no_inputs = no_inputs
        self._no_outputs = no_outputs
        
        assert len(pose_traj.keys()) == len(adj_traj.keys()), \
        "The number of rows should be the same for poses and flat_adj."
        
        # Check that the traj lens are all equal
        assert self.check_traj_len(pose_traj), \
            "The length of the pose trajectories are not equal."
            
        assert self.check_traj_len(adj_traj), \
            "The length of the adj trajectories are not equal."       
        
        self._no_keys = len(pose_traj.keys())
        self._traj_len = pose_traj[0].shape[0]
        
        # Create data dict
        self._data = {'poses': pose_traj,
                        'flat_adj': adj_traj}
        
        self._no_nodes = int(np.sqrt(adj_traj[0].shape[1]))
        
    def check_traj_len(self, data_dict: Dict):
        traj_lens = [0] * len(data_dict.keys())
        for k, item in data_dict.items():
            traj_lens[k] = item.shape[0]
        
        return all(x == traj_lens[0] for x in traj_lens)   
    
    def get_index(self, index: int) -> Tuple[int, int]:
        key_index = index // (self._traj_len - self._no_inputs - self._no_outputs)
        traj_index = index % (self._traj_len - self._no_inputs - self._no_outputs)       
        return key_index, traj_index      
                
    def __len__(self):
        return self._no_keys * (self._traj_len - self._no_inputs - self._no_outputs)
    
    def __getitem__(self, index: int):
        key_index, traj_index = self.get_index(index)
        
        x_in = self._data['poses'][key_index][traj_index: traj_index + self._no_inputs, :]
        x_out = self._data['poses'][key_index][traj_index + self._no_inputs: traj_index + self._no_inputs + self._no_outputs, :]
        
        adj_mat = self._data['flat_adj'][key_index][traj_index + self._no_inputs].reshape(self._no_nodes, self._no_nodes)
        return x_in, x_out, adj_mat
            
            

# if __name__ == "__main__":
#     path = "data/graphs/"
    
#     process_data = ProcessTrajectories(np.array([0.6, 0.6, 0.08]))
#     poses, adj = process_data(path, ["l_beam_1", "l_beam_2", "l_pin_A"])
    
#     # Data loader
#     dataset = TrajectoryDataset(poses, adj, torch.device('cuda'))
    
#     x_in, x_out, adj_mat = dataset.__getitem__(1)
    
#     import pdb
#     pdb.set_trace()