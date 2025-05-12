import time
import os
import copy

import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer
import networkx as nx

from beam_data_gen.beam_impl.robot_graph import (square_graphs, RobotGraph)
from beam_data_gen.data_sampling.beam_sampler import BeamSampler
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
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for name, graph in graphs.items():
            # Create the datasaver
            datasaver = DataSaver(graph)
            
            for left_connection in left_connections:
                # Add hand connections
                graph.add_hand("robot_left_hand", left_connection)
                
                for right_connection in right_connections:
                    
                    counter = 0
                    max_iters = 5
            
                    # Add hand connections
                    graph.add_hand("robot_right_hand", right_connection)
                                    
                    # Start loop and sample pose
                    while viewer.is_running() and counter < max_iters:
                        step_start = time.time()
                        
                        # mj_step can be replaced with code that also evaluates
                        # a policy and applies a control signal before stepping the physics.
                        mujoco.mj_step(m, d)

                        # Pick up changes to the physics state, apply perturbations, update options from GUI.
                        viewer.sync()
                        
                        print(left_connection, right_connection, name)
                        
                        # Save data
                        collision, nodes = check_graph_collisions(d, graph)
                        if not collision:
                            datasaver.append_graph(graph)
                            counter += 1
                        
                        else:
                            print(f"In collision, nodes: {nodes[0]}, {nodes[1]}")
                        
                        # Resample a pose
                        sampler.sample_poses(graph, sampler.uniform_pose_sampler)
                        pose_dict = sampler.graph_to_pose_dict(graph)
                        set_q(d.qpos, pose_dict)  
                        
                        # Rudimentary time keeping, will drift relative to wall clock.
                        time_until_next_step = m.opt.timestep - (time.time() - step_start)
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)
                        
                    
            # Save data
            path_dir = os.path.join("data/robot_square_0")
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            datasaver.df.to_pickle(os.path.join(path_dir, str(name) + ".pkl"))           

    return 0


if __name__ == "__main__":
    main()
