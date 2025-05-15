from typing import List
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from matplotlib import pyplot as plt
import mujoco
import mujoco.viewer

from beam_data_gen.beam_impl.Square_graph import square_connected_graph, RampGraph
from beam_data_gen.models.datasets.process_data import ProcessData
from beam_data_gen.data_sampling.beam_sampler import BeamSampler
from beam_data_gen.simulator.square_robot_sim import SquareRobotSim


def graph_to_pose(graph: RampGraph, node_names: List[str], data_processor: ProcessData):
    no_nodes = len(node_names)    
    pose_target = np.zeros(no_nodes * data_processor.state_dim)
    for k, name in enumerate(node_names):
        data = graph.graph.nodes[name]
        pose_target[data_processor.state_dim * k: data_processor.state_dim * (k + 1)] = data_processor.pose_to_rep(data['pose'])
    return pose_target


def normalise_pose(pose_torch: torch.tensor, state_dim: int):
    no_items = pose_torch.shape[0] // state_dim    
    for k in range(no_items):
        pose_torch[state_dim * k + 3: state_dim * k + 5] = torch.nn.functional.normalize(pose_torch[state_dim * k + 3: state_dim * k + 5], dim=0)
    return pose_torch


def main():
    device = torch.device("cuda")
    # Data processor
    process_data = ProcessData(np.array([1.0, 1.0, 1.0]))  
    # Simulator
    sim = SquareRobotSim(process_data)  
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
    trans_lims = [0.60, 0.60, 0.0]
    sampler = BeamSampler(trans_lims)
    # Remove all edges and perturb
    graph.A = np.zeros([no_nodes, no_nodes])
    sampler.sample_poses(graph, sampler.uniform_pose_sampler)
    
    # 
    pose_init = graph_to_pose(graph, node_names, process_data)
    print(f"Pose init: {pose_init}")
    
    pose_init_torch = torch.tensor(pose_init, dtype=torch.float32).to(device)
    pose_torch = pose_init_torch.requires_grad_(True)
    
    # Define losses
    loss_func = nn.MSELoss(reduction="sum")
    alpha = 1.0e-2
    no_iters = 250
    
    pose_lst = []
    loss_lst = []
    
    for _ in range(no_iters):
        loss =  loss_func(pose_torch, pose_tar_torch)
        pose_torch = pose_torch - alpha * grad(outputs=loss, inputs=pose_torch, retain_graph=True)[0]
        pose_torch = normalise_pose(pose_torch, state_dim=process_data.state_dim)
        loss_lst.append(loss.clone().detach().cpu().numpy())
        
        pose_lst.append(pose_torch)
    
    # Beam Trajectories
    beam_traj = torch.stack(pose_lst, dim=0)    
    beam_traj = torch.cat([torch.zeros([beam_traj.shape[0], 2 * process_data.state_dim], dtype=torch.float32).to(device), beam_traj], 1)
    
    print(f"Final pose: {pose_torch.detach().cpu().numpy()}")
    print(f"Pose target: {pose_target}")
    print(f"Loss: {loss.detach().cpu().numpy()}")
    
    plt.figure()
    plt.plot(loss_lst)
    plt.show()
    
    # Visualise results
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_square.xml')
    d = mujoco.MjData(m)
    
    # Visualisation runs
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        input("continue")
        
        # Start loop and sample pose
        while viewer.is_running():
            for k in range(beam_traj.shape[0]):
                
                 # Decoder the prediction
                sim.decode_x(d, beam_traj[k:k+1, :])
                                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()   
                
                # Rudimentary time keeping, will drift relative to wall clock.
                time.sleep(0.05)
        
    
    return 0    

if __name__ == "__main__":
    main()
