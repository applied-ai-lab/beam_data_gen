import time
import os
import copy

import pandas as pd
import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer
import networkx as nx

from beam_data_gen.beam_impl.robot_graph import (square_graphs, RobotGraph)
from beam_data_gen.data_sampling.beam_sampler import BeamSampler, PoseSamplerParams
from beam_data_gen.data_sampling.data_saver import DataSaver


def pose_to_q(trans, rot):
    pose = np.zeros(7)
    pose[0:3] = trans
    # Mujoco quat order w x y z
    pose[3] = rot.as_quat()[-1]
    pose[4:] = rot.as_quat()[0:-1]
    return pose 


def set_q(q_pos, q_pos_dict):
    # Ramp components
    q_pos[0:7] = pose_to_q(q_pos_dict["square_beam_1"].trans, q_pos_dict["square_beam_1"].orient)
    q_pos[7:14] = pose_to_q(q_pos_dict["square_pin_A"].trans, q_pos_dict["square_pin_A"].orient)
    q_pos[14:21] = pose_to_q(q_pos_dict["square_beam_2"].trans, q_pos_dict["square_beam_2"].orient)
    q_pos[21:28] = pose_to_q(q_pos_dict["square_pin_B"].trans, q_pos_dict["square_pin_B"].orient)
    q_pos[28:35] = pose_to_q(q_pos_dict["square_beam_3"].trans, q_pos_dict["square_beam_3"].orient)
    q_pos[35:42] = pose_to_q(q_pos_dict["square_pin_C"].trans, q_pos_dict["square_pin_C"].orient)
    q_pos[42:49] = pose_to_q(q_pos_dict["square_beam_4"].trans, q_pos_dict["square_beam_4"].orient)
    q_pos[49:56] = pose_to_q(q_pos_dict["square_pin_D"].trans, q_pos_dict["square_pin_D"].orient)
    # Hands
    q_pos[56:63] = pose_to_q(q_pos_dict["robot_left_hand"].trans, q_pos_dict["robot_left_hand"].orient)
    q_pos[63:70] = pose_to_q(q_pos_dict["robot_right_hand"].trans, q_pos_dict["robot_right_hand"].orient)
    return

    
def check_graph_collisions(data, ramp_graph: RobotGraph):
    
    # Step 1: Compute connected components
    components = list(nx.connected_components(ramp_graph.graph))

    # Step 2: Create a mapping from node to component ID
    node_to_component = {}
    for idx, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = idx

    # Step 3: Function to check if two nodes are in the same component
    def same_component(node1, node2):
        return node_to_component.get(node1) == node_to_component.get(node2)
    
    collision = False
    geom_to_name = {1: "square_beam_1",
                    2: "square_pin_A",
                    3: "square_beam_2",
                    4: "square_pin_B",
                    5: "square_beam_3",
                    6: "square_pin_C",
                    7: "square_beam_4",
                    8: "square_pin_D"}
    for i in range(data.ncon):  # Iterate through contacts
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        if geom1 not in geom_to_name.keys() or geom2 not in geom_to_name.keys():
            continue
        # If there is a contact between two items which are not connected return true
        elif not same_component(geom_to_name[geom1], geom_to_name[geom2]):
            return True, [geom_to_name[geom1], geom_to_name[geom2]]
    return collision, []



def main():

    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_square.xml')
    d = mujoco.MjData(m)

    # Initialise the classes
    trans_lims = [0.30, 0.30, 0.0]
    sampler = BeamSampler(trans_lims)

    # Beam config graph
    graphs = square_graphs
    
    # Hand connections
    left_connections = [None,
                        "square_beam_1",
                        "square_pin_A",
                        "square_beam_2",
                        "square_pin_B",
                        "square_beam_3",
                        "square_pin_C",
                        "square_beam_4",
                        "square_pin_D"]
    right_connections = copy.deepcopy(left_connections)
    
    # Set up parameters
    dt = 0.2
    duration = 10.0
    max_velocities = np.array([0.25, 0.25, 0.05, 0.25, 0.25, 0.25])
    params = PoseSamplerParams(dt, duration, None, max_velocities, np.array([1, 1, 1, 0, 0, 1]))
    
    no_random_inits = 5
    
    datasaver_dict = {}
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for name, graph in graphs.items():    
            
            hand_idx = 0        
            
            for left_connection in left_connections:
                # Add hand connections
                graph.add_hand("robot_left_hand", left_connection)
                
                for right_connection in right_connections:
                                
                    # Add hand connections
                    graph.add_hand("robot_right_hand", right_connection)
                    
                    for _ in range(no_random_inits):
                        print(left_connection, right_connection, name)
                        
                        # Find trajectories
                        sampler.move_hands(graph, params)
                        traj_counter = 0
                        
                        counter = 0
                        
                        # Create the datasaver
                        datasaver = DataSaver(graph)
                                        
                        # Start loop and sample pose
                        while viewer.is_running() and counter < params.no_samples:
                            
                            # mj_step can be replaced with code that also evaluates
                            # a policy and applies a control signal before stepping the physics.
                            mujoco.mj_step(m, d)

                            # Pick up changes to the physics state, apply perturbations, update options from GUI.
                            viewer.sync()
                            
                            if traj_counter >= params.no_samples:
                                sampler.move_hands(graph, params)
                                traj_counter = 0
                            
                            # Sample a trajectory given the pose
                            sampler.set_pose_with_traj(graph, traj_counter)
                            
                            pose_dict = sampler.graph_to_pose_dict(graph)
                            set_q(d.qpos, pose_dict)  
                            
                            # Save data if nto in collision
                            collision, nodes = check_graph_collisions(d, graph)
                            if not collision:
                                # Data saver append graph
                                datasaver.append_graph(graph)
                                counter += 1
                            
                            else:                
                                print(f"In collision, nodes: {nodes[0]} and {nodes[1]}")
                                print(left_connection, right_connection, name)
                                    
                            traj_counter += 1
                            
                            # time.sleep(dt)
                            
                        datasaver_dict[hand_idx] = [copy.deepcopy(datasaver.df)]
                        hand_idx += 1
            
            data_df = pd.DataFrame.from_dict(datasaver_dict, orient="index")
            # Save data
            path_dir = os.path.join("data/trajectories_square_small_1")
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            data_df.to_pickle(os.path.join(path_dir, str(name) + ".pkl"))           

    return 0


if __name__ == "__main__":
    main()
