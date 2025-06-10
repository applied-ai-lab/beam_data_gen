from typing import List
import time
from enum import Enum
    

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from matplotlib import pyplot as plt
import mujoco
import mujoco.viewer
from tqdm import trange

from beam_data_gen.beam_impl.Square_graph import square_connected_graph, RampGraph
from beam_data_gen.models.datasets.process_data import ProcessData
from beam_data_gen.data_sampling.beam_sampler import BeamSampler
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim
from beam_data_gen.traj_opt.dual_assembly import DualAssembly, TrajOptParams


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
    torch.manual_seed(seed)
    # Set device
    device = torch.device("cuda")
    # Data processor
    process_data = ProcessData(np.array([1.0, 1.0, 1.0]))  
    # Simulator
    sim = SquareRobotSim(process_data)  
    # Hands
    left_pose = torch.tensor([0.15, 0.55, 0.25, 0.0, 0.0], dtype=torch.float32).requires_grad_(True).to(device)
    right_pose = torch.tensor([0.00, 0.00, 0.25, 0.0, 0.0], dtype=torch.float32).requires_grad_(True).to(device)
    
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
    # Define the graph (ignore the hands for now)
    graph: RampGraph = square_connected_graph
    # Find the target
    pose_target = graph_to_pose(graph, node_names, process_data)
    print(f"Pose target: {pose_target}")
    
    pose_tar_torch = torch.tensor(pose_target, dtype=torch.float32).to(device)
    
    # Find initial condition
    trans_lims = [1.0, 1.0, 0.0]
    sampler = BeamSampler(trans_lims)
    # Remove all edges and perturb
    graph.A = np.zeros([no_nodes, no_nodes])
    sampler.sample_poses(graph, sampler.uniform_pose_sampler)
    
    # 
    pose_init = graph_to_pose(graph, node_names, process_data)
    print(f"Pose init: {pose_init}")
    
    pose_init_torch = torch.tensor(pose_init, dtype=torch.float32).to(device)
    pose_torch = pose_init_torch.requires_grad_(True)
    
    # Simulate results
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_square.xml')
    d = mujoco.MjData(m)
    
    params = TrajOptParams(step_size=0.01,
                            no_steps=50,
                            epsilon=1.e-2,
                            no_particles=30)
    
    traj_opt = DualAssembly(params, state_dim=process_data.state_dim, sim=sim)
    
    # Set the values
    traj_opt.set_x(left_pose, right_pose, pose_init_torch)
    traj_opt.goal = pose_tar_torch
    
    # Optimise
    particles = traj_opt.optimise(m, d)
    
    plt.figure()
    plt.plot(particles.no_live_particles)
    plt.show()
    
    trajectory, indices = particles.sample_trajectory()
    
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
                time.sleep(0.5)
    
    return 0    

if __name__ == "__main__":
    main()
