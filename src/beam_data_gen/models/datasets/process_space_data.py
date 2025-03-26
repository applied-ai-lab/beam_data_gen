from typing import List
import os
import copy

import numpy as np 
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R



class ProcessSpaceData:
    def __init__(self, pos_lims: np.array):
        self._pos_lims = pos_lims
        
    def load_data(self, dir_path: str) -> List[pd.DataFrame]:
        pd_files = []        
        for root, _, files in os.walk(dir_path):
            pd_files = list(pd.read_pickle(os.path.join(root, file)) for file in files)  
        return pd_files   
    
    def extract_pose(self, data: np.array):
        pos = data[:, 0:3]
        quat = data[:, 3:]
        
        rot = R.from_quat(quat)
        # Theta z
        theta_z = rot.as_euler(seq="xyz")[:, 2]
        return np.hstack([pos, np.sin(theta_z).reshape(-1, 1), np.cos(theta_z).reshape(-1, 1)])
    
    def denorm_output(self, x_pred: torch.tensor):
        state_dim = 5
        
        pos_lim = torch.tensor(self._pos_lims, dtype=x_pred.dtype, device=x_pred.device)
        
        x_denorm = x_pred.detach().clone()
        for k in range(int(x_pred.shape[1] / state_dim)):
            
            x_denorm[:, state_dim*k : state_dim*k + 3] *= pos_lim
            # x_denorm[:, state_dim*k + 3] = torch.atan2(x_pred[:, state_dim*k + 3].min(torch.ones([1], dtype=x_pred.dtype, device=x_pred.device)).max(-torch.ones([1], dtype=x_pred.dtype, device=x_pred.device)), x_pred[:, state_dim*k + 4].min(1.0).max(-1.0))
            x_denorm[:, state_dim*k + 3] = torch.atan2(x_pred[:, state_dim*k + 3], x_pred[:, state_dim*k + 4])
            x_denorm[:, state_dim*k + 4] = torch.atan2(x_pred[:, state_dim*k + 3], x_pred[:, state_dim*k + 4])
            
        return x_denorm        
    
    def extract_data(self, dir_path, beam_names):
        pd_file_lst = self.load_data(dir_path)
        
        pose_lst = []
        adj_lst = []
        connectivity_lst = []

        for pd in pd_file_lst:
            data_lst = []
            for k, beam in enumerate(beam_names):
                pose_data = self.extract_pose(np.vstack(pd[beam].to_list()))
                
                # Standardise position
                pose_standardised = copy.deepcopy(pose_data)
                pose_standardised[:, 0:3] = np.divide(pose_data[:, 0:3], self._pos_lims) 
                
                data_lst.append(pose_standardised)
            
            pose_mat = np.hstack(data_lst)
            flat_adj_mat = np.array(pd["adj_mat"].to_list())
            
            # No Nodes
            no_nodes = int(np.sqrt(flat_adj_mat.shape[1]))         
            left_connect = flat_adj_mat[:, 0:no_nodes]
            right_connect = flat_adj_mat[:, no_nodes: 2*no_nodes]
            
            # Stack connections along 3rd axis
            connections = np.stack([left_connect, right_connect], axis=2)
            row_sums = connections.sum(axis=1).squeeze()
            result = (row_sums > 0).astype(int)
                        
            # Store matrices in a list
            pose_lst.append(pose_mat)
            adj_lst.append(flat_adj_mat)
            connectivity_lst.append(result)
        
        poses = np.vstack(pose_lst)
        adj = np.vstack(adj_lst)
        connectivity = np.vstack(connectivity_lst)
        return poses, adj, connectivity
    
    def __call__(self, dir_path, beam_names):
        return self.extract_data(dir_path, beam_names)