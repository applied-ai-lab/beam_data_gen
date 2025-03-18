from typing import List
import os
import copy

import numpy as np 
import pandas as pd
import torch

from beam_data_gen.models.datasets.process_data import ProcessData


class ProcessTrajectories(ProcessData):
    def __init__(self, pos_lims, device: torch.device):
        super().__init__(pos_lims)
        self._device = device
        
    def extract_data(self, dir_path, beam_names):
        pd_file_lst = self.load_data(dir_path)
        
        pose_dict = {}
        adj_dict = {}
        
        counter = 0

        for graphs_indices, graph_df in enumerate(pd_file_lst):
            # Iterate through the rows of each graph            
            for hand_connection_index, trajectory_lst in graph_df.iterrows():
                data_lst = []
                # Extract data using beam names          
                for _, beam in enumerate(beam_names):
                    pose_data = self.extract_pose(np.vstack(trajectory_lst[0][beam].to_list()))
                    
                    # Standardise position
                    pose_standardised = copy.deepcopy(pose_data)
                    pose_standardised[:, 0:3] = np.divide(pose_data[:, 0:3], self._pos_lims)
                    
                    data_lst.append(pose_standardised)
            
                pose_mat = np.hstack(data_lst)
                flat_adj_mat = np.array(trajectory_lst[0]["adj_mat"].to_list())
                
                # Store matrices in a dictionary
                pose_dict[counter] = torch.tensor(pose_mat, dtype=torch.float32).to(self._device)
                adj_dict[counter]  = torch.tensor(flat_adj_mat, dtype=torch.float32).to(self._device)
                
                counter += 1
        
        return pose_dict, adj_dict
    
    def __call__(self, dir_path, beam_names):
        return self.extract_data(dir_path, beam_names)

