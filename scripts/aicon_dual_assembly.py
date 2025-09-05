from typing import List
import time
from enum import Enum
import os

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
    torch.manual_seed(seed)
    # Set device
    device = torch.device("cpu")
    # Data processor
    process_data = ProcessData(np.array([1.0, 1.0, 1.0]))  
    # Hands
    left_pose = torch.tensor([0.15, 0.25, 0.25, 0.0, 0.0], dtype=torch.float32).requires_grad_(True).to(device)
    right_pose = torch.tensor([0.00, -0.25, 0.25, 0.0, 0.0], dtype=torch.float32).requires_grad_(True).to(device)
    
    # No. runs
    no_runs = 40
    
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
    
    params = TrajOptParams(step_size=0.2,
                            no_steps=180,
                            epsilon=1.e-3,
                            no_particles=1, 
                            device=device)
    
    state_params = StateParams(state_dim=5,
                                no_beams=no_nodes,
                                no_hands=2,
                                no_pins=4,
                                device=device,
                                tol=1.0e-4)
    
    traj_opt = DualAssembly(params, 
                            state_params=state_params, 
                            sim=sim,
                            left_start=left_pose.clone(),
                            right_start=right_pose.clone(),
                            model=m, data=d)
    
    # Beam losses
    beam_loss = np.zeros((no_runs, params.no_steps))
    
    for k in trange(no_runs):
        
        seed += 1
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        sampler.sample_poses(graph, sampler.uniform_pose_sampler)
        pose_init = graph_to_pose(graph, node_names, process_data)
        pose_init_torch = torch.tensor(pose_init, dtype=torch.float32).to(device).requires_grad_(True)
        
        # Set the values
        states = DualArmStates(state_params)
        states.beam_poses = pose_init_torch
        states.left_pose = left_pose.clone()
        states.right_pose = right_pose.clone()
        states.left_start=left_pose.clone()
        states.right_start=right_pose.clone()
        states.beam_goal = pose_tar_torch
        states.initialise()
        
        # Set the states
        traj_opt.states = states
        
        # Optimise
        particles = traj_opt.optimise()   
        
        # Beam loss
        beam_loss[k, :] = particles.loss.cpu().detach().clone().numpy()  
        
        # Save beam losses
        np.save(os.path.join("data", "beam_losses", "beam_loss.npy"), beam_loss)
    
    plt.figure()
    plt.plot(beam_loss.transpose()[1:, :])
    
    import pdb
    pdb.set_trace()
    
    plt.figure()
    plt.plot(particles.no_live_particles)
    plt.show()
    
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
