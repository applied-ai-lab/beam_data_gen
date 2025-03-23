import os

import numpy as np
import pandas as pd 
import torch

from beam_data_gen.models.datasets.process_trajectories import ProcessTrajectories
from beam_data_gen.models.datasets.trajectory_dataset import TrajectoryDataset


def extract_trajectory_segment(df, index, start, N):
    """
    Extract a short sequence of trajectory poses of length N from a nested DataFrame.
    
    Parameters:
    df (pd.DataFrame): The main DataFrame containing the nested trajectory DataFrame.
    index: The index in the main DataFrame where the trajectory is stored.
    start (int): The starting index in the trajectory DataFrame.
    N (int): The length of the sequence to extract.
    
    Returns:
    pd.DataFrame: The extracted trajectory segment.
    """
    if index not in df.index:
        raise ValueError("Provided index is not in the main DataFrame.")
    
    trajectory_df = df.loc[index, 'trajectory']  # Assuming the column is named 'trajectory'
    
    if not isinstance(trajectory_df, pd.DataFrame):
        raise ValueError("Expected a DataFrame in the 'trajectory' column.")
    
    return trajectory_df.iloc[start:start + N]


def main():
    
    path = os.path.join("data/trajectories_big_1")
    
    no_inputs = 5
    no_outputs = 7
    device = torch.device("cuda")
    
    traj_processor = ProcessTrajectories(np.array([0.6, 0.6, 0.08]), device) 
    
    pose_traj, adj_traj = traj_processor.extract_data(path, ["l_beam_1", "l_beam_2", "l_pin_A"])
    dataset = TrajectoryDataset(pose_traj, adj_traj, no_inputs, no_outputs)
    
    print(dataset.__len__())
    
    x_in, x_out, adj_mat = dataset.__getitem__(0)
    
    import pdb
    pdb.set_trace()
    
    return 0


if __name__ == "__main__":
    main()
