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


def max_gradient(counter, gradient, state_dim):
    # Pin penalty
    pin_indices = list(2 * k * state_dim + i + state_dim for k in range(4) for i in range(state_dim))    
    gradient[pin_indices] *= 2.0
    
    gradient_reshaped = gradient.view(-1, state_dim)
    grad_norm = torch.norm(gradient_reshaped, p=2.0, dim=1)
    # Item with largest gradient
    max_idx = torch.argmin(grad_norm)
    min_val = grad_norm[max_idx]
    while min_val < 0.01:
        counter += 1
        if counter >= (gradient.shape[0] // state_dim):
            gradient *= 0.0
            break
        
        gradient_reshaped[max_idx, :] *= 1.0e6
        grad_norm = torch.norm(gradient_reshaped, p=2.0, dim=1)
        max_idx = torch.argmin(grad_norm)
        min_val = grad_norm[max_idx]        
        
    grad_mask = torch.zeros_like(gradient, dtype=gradient.dtype).to(gradient.device)
    grad_mask[state_dim * max_idx: state_dim * max_idx + state_dim] = torch.ones((state_dim), dtype=gradient.dtype).to(gradient.device)    
    return grad_mask * gradient
    

def estimate_contact(hand_pose: torch.tensor, beam_pose: torch.tensor):
    loss_func = nn.MSELoss(reduction="sum")
    return loss_func(hand_pose, beam_pose)


def pred_contact(hand_pose: torch.tensor, beam_pose: torch.tensor, tol=1.0e-3):
    error = estimate_contact(hand_pose, beam_pose)
    return torch.abs(error) < tol


def find_small_index(gradients, dim):
    grad_norm = torch.norm(gradients, p=2.0, dim=dim)
    return torch.argmin(grad_norm)


def calc_losses(counter, beam_poses, beam_targets, left_hand, right_hand, tol=1.0e-2):
    state_dim = 5
    # Pin penalty
    pin_indices = list(2 * k for k in range(4))
    
    # We want the loss per beam or per hand
    beam_loss = nn.MSELoss(reduction="sum")
    hand_loss = nn.MSELoss(reduction='none')
        
    beam_losses = beam_loss(beam_poses, beam_targets)
    
    left_loss = hand_loss(beam_poses, left_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
    right_loss = hand_loss(beam_poses, right_hand.repeat(beam_poses.shape[0], 1)).sum(dim=1)
    
    # Beam gradients
    beam_gradients = grad(outputs=beam_losses, inputs=beam_poses, retain_graph=True)[0]
    
    gradient_reshaped = beam_gradients.view(-1, state_dim)
    grad_norm = torch.norm(gradient_reshaped, p=2.0, dim=1)
    # Item with largest gradient
    max_idx = torch.argmin(grad_norm)
    min_val = grad_norm[max_idx]
    while min_val < 0.01:
        counter += 1
        if counter >= beam_gradients.shape[0]:
            beam_gradients *= 0.0
            break
        
        gradient_reshaped[max_idx, :] *= 1.0e6
        left_loss[max_idx] *= 1.0e6
        right_loss[max_idx] *= 1.0e6      
        
        grad_norm = torch.norm(gradient_reshaped, p=2.0, dim=1)
        max_idx = torch.argmin(grad_norm)
        min_val = grad_norm[max_idx]
        
    # Use loss to calculate contacts
    left_contacts = (left_loss < tol).type(torch.float32)
    right_contacts = (right_loss < tol).type(torch.float32)
    
    left_loss[pin_indices] *= 2.0
    right_loss[pin_indices] *= 2.0
    
    # Find smallest gradients
    right_index = torch.argmin(right_loss, 0)
    left_index = torch.argmin(left_loss, 0)
    
    if right_index == left_index:
        right_loss[right_index] = 1.0e3
        right_index = torch.argmin(right_loss, 0)
    
    beam_gradients = (beam_gradients * left_contacts.reshape(beam_gradients.shape[0], 1) + beam_gradients * right_contacts.reshape(beam_gradients.shape[0], 1))
    
    # Hand gradients
    left_gradients = grad(left_loss[left_index], inputs=left_hand, retain_graph=True)[0] * (1. - left_contacts[left_index]) + beam_gradients[left_index, :]
    right_gradients = grad(right_loss[right_index], inputs=right_hand, retain_graph=True)[0] * (1. - right_contacts[right_index]) + beam_gradients[right_index, :]
    
    return beam_gradients, left_gradients, right_gradients    


def main():
    # Set seeds
    seed = 42
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
    
    # Define losses
    alpha = 1.0e-1
    no_iters = 200
    
    pose_lst = []
    
    left_lst = []
    right_lst = []
    
    beam_grad_lst = []
    left_grad_lst = []
    right_grad_lst = []
    
    counter = 0
    
    for _ in range(no_iters):
        
        beam_grads, left_grad, right_grad = calc_losses(counter,
                                                        pose_torch.view(-1, 5),
                                                        pose_tar_torch.view(-1, 5),
                                                        left_pose,
                                                        right_pose)
        left_pose = left_pose - alpha * left_grad
        right_pose = right_pose - alpha * right_grad
        # Calc gradient
        
        pose_torch = pose_torch - alpha * beam_grads.view(-1)
        pose_torch = normalise_pose(pose_torch, state_dim=process_data.state_dim)
        left_pose = normalise_pose(left_pose, state_dim=process_data.state_dim)
        right_pose = normalise_pose(right_pose, state_dim=process_data.state_dim)
        
        pose_lst.append(pose_torch)
        left_lst.append(left_pose)
        right_lst.append(right_pose)
        
        beam_grad_lst.append(torch.norm(beam_grads, p=2.0))
        left_grad_lst.append(torch.norm(left_grad, p=2.0))
        right_grad_lst.append(torch.norm(right_grad, p=2.0))
    
    # Beam Trajectories
    beam_traj = torch.stack(pose_lst, dim=0)
    
    left_traj = torch.stack(left_lst, dim=0)
    right_traj = torch.stack(right_lst, dim=0)
    
    beam_grad_traj = torch.stack(beam_grad_lst, dim=0)
    left_grad_traj = torch.stack(left_grad_lst, dim=0)
    right_grad_traj = torch.stack(right_grad_lst, dim=0)
    
    beam_traj = torch.cat([left_traj, right_traj, beam_traj], 1)
    
    print(f"Final pose: {pose_torch.detach().cpu().numpy()}")
    print(f"Pose target: {pose_target}")
    # print(f"Loss: {loss.detach().cpu().numpy()}")
    
    plt.figure()
    plt.plot(beam_grad_traj.detach().cpu().numpy())
    plt.plot(left_grad_traj.detach().cpu().numpy())
    plt.plot(right_grad_traj.detach().cpu().numpy())
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
                time.sleep(0.1)
        
    
    return 0    

if __name__ == "__main__":
    main()
