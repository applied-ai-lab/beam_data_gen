from typing import List
import time
from enum import Enum
import json
import copy
    
import os

import numpy as np
from matplotlib import pyplot as plt
import mujoco
import mujoco.viewer
from tqdm import trange
from scipy.spatial.transform import Rotation as R


from beam_data_gen.beam_impl.Square_graph import square_connected_graph, RampGraph
from beam_data_gen.models.datasets.process_data import ProcessData
from beam_data_gen.data_sampling.beam_sampler import BeamSampler
from beam_data_gen.simulator.sim_robot import SimRobot
from beam_data_gen.traj_opt.dual_assembly import DualAssembly, TrajOptParams, StateParams, DualArmStates


def graph_to_pose(graph: RampGraph, node_names: List[str], data_processor: ProcessData):
    no_nodes = len(node_names)    
    pose_target = np.zeros(no_nodes * data_processor.state_dim)
    for k, name in enumerate(node_names):
        data = graph.graph.nodes[name]
        pose_target[data_processor.state_dim * k: data_processor.state_dim * (k + 1)] = data_processor.pose_to_rep(data['pose'])
    return pose_target


def main():
    # Set seeds
    seed = 1000
    np.random.seed(seed)

    # Data processor
    process_data = ProcessData(np.array([1.0, 1.0, 1.0]))  
    # Hands
    left_pose = np.array([0.15, 0.25, 0.25, 0.0, 0.0])
    right_pose = np.array([0.00, -0.25, 0.25, 0.0, 0.0])
    
    # Node names for consideration
    node_names = ["square_beam_1",
                    "square_pin_A",
                    "square_beam_2",
                    "square_pin_B",
                    "square_beam_3",
                    "square_pin_C",
                    "square_beam_4",
                    "square_pin_D"]
    
    no_nodes = len(node_names)
    # Simulator
    sim = SimRobot(process_data, node_names)  
    # Define the graph (ignore the hands for now)
    graph: RampGraph = square_connected_graph
    # Find the target
    pose_target = graph_to_pose(graph, node_names, process_data)
    print(f"Pose target: {pose_target}")
    
    # Find initial condition
    trans_lims = [0.2, 0.2, 0.0]
    sampler = BeamSampler(trans_lims)
    # Remove all edges and perturb
    graph.A = np.zeros([no_nodes, no_nodes])
    sampler.sample_poses(graph, sampler.uniform_pose_sampler)
    
    # Calc initial state
    pose_init = graph_to_pose(graph, node_names, process_data)
    print(f"Pose init: {pose_init}")
    
    # Simulate results
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_square.xml')
    d = mujoco.MjData(m)

    
    
    indices = particles.sample_indices()
    trajectory, gripper_states = particles.sample_trajectories(indices)
    trajectory = trajectory[:, 0:(state_params.no_hands + state_params.no_beams) * state_params.state_dim]
    
    # Visualisation runs
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        input("continue")
        
        # Start loop and sample pose
        while viewer.is_running():
            for k in trange(trajectory.shape[0]):
                
                # Decoder the prediction
                sim.decode_x(d, trajectory[k:k+1, :])
                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()   
                
                # Rudimentary time keeping, will drift relative to wall clock.
                time.sleep(0.2)
    
    return 0    

if __name__ == "__main__":
    main()
