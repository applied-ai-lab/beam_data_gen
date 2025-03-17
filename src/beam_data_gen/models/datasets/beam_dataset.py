import os
import copy
from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from beam_data_gen.models.datasets.process_data import ProcessData
        
            
class BeamDataset(Dataset):
    def __init__(self, 
                 poses: np.array, 
                 flat_adj: np.array,
                 device: torch.device):
        super().__init__()
        
        assert poses.shape[0] == flat_adj.shape[0], \
        "The number of rows should be the same for poses and flat_adj."
        
        self._data = {'poses': torch.tensor(poses, dtype=torch.float32, device=device),
                      'flat_adj': torch.tensor(flat_adj, dtype=torch.float32, device=device)}
        
        self._no_nodes = int(np.sqrt(flat_adj.shape[1]))
        
    def __len__(self):
        return self._data["poses"].shape[0]
    
    def __getitem__(self, index):
        x_in = self._data['poses'][index, :]
        x_out = self._data['poses'][index, :]
        
        adj_mat = self._data['flat_adj'][index, :].reshape(self._no_nodes, self._no_nodes)
        return x_in, x_out, adj_mat
            
            
    

if __name__ == "__main__":
    path = "data/graphs/"
    
    process_data = ProcessData(np.array([0.6, 0.6, 0.08]))
    poses, adj = process_data(path, ["l_beam_1", "l_beam_2", "l_pin_A"])
    
    # Data loader
    dataset = BeamDataset(poses, adj, torch.device('cuda'))
    
    x_in, x_out, adj_mat = dataset.__getitem__(1)
    
    import pdb
    pdb.set_trace()