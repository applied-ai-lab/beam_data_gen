import time
import os
import copy

import pandas as pd
import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer

from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot, RobotGraph)
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
    q_pos[0:7] = pose_to_q(q_pos_dict["l_beam_1"].trans, q_pos_dict["l_beam_1"].orient)
    q_pos[7:14] = pose_to_q(q_pos_dict["l_beam_2"].trans, q_pos_dict["l_beam_2"].orient)
    q_pos[14:21] = pose_to_q(q_pos_dict["l_pin_A"].trans, q_pos_dict["l_pin_A"].orient)
    q_pos[21:28] = pose_to_q(q_pos_dict["robot_left_hand"].trans, q_pos_dict["robot_left_hand"].orient)
    q_pos[28:35] = pose_to_q(q_pos_dict["robot_right_hand"].trans, q_pos_dict["robot_right_hand"].orient)
    return

    
def check_graph_collisions(data, ramp_graph: RobotGraph):
    collision = False
    geom_to_name = {1: "l_beam_1",
                    2: "l_beam_2",
                    3: "l_pin_A"}
    for i in range(data.ncon):  # Iterate through contacts
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        if geom1 not in geom_to_name.keys() or geom2 not in geom_to_name.keys():
            continue
        # If there is a contact between two items which are not connected return true
        elif not ramp_graph.graph.has_edge(geom_to_name[geom1], geom_to_name[geom2]):
            return True
    return collision
          


def main():

    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)

    # Initialise the classes
    trans_lims = [0.0, 0.0, 0.0]
    sampler = BeamSampler(trans_lims)

    # Beam config graph
    graphs = {"connected": l_connected_robot, 
              "pin_removed": l_pin_removed_robot, 
              "disconnected": l_disconnected_robot}
    
    # Hand connections
    left_connections = [None,
                        "l_beam_1",
                        "l_beam_2",
                        "l_pin_A"]
    right_connections = copy.deepcopy(left_connections)
    
    # Set up parameters
    dt = 0.2
    duration = 10.0
    max_velocities = np.array([0.25, 0.25, 0.05, 0.25, 0.25, 0.25])
    params = PoseSamplerParams(dt, duration, None, max_velocities, np.array([1, 1, 1, 0, 0, 1]))
    
    no_random_inits = 500
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for name, graph in graphs.items():    
            
            # Create the datasaver
            datasaver = DataSaver(graph)
            
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
                                        
                        # Start loop and sample pose
                        while viewer.is_running() and traj_counter < params.no_samples:
                            
                            # mj_step can be replaced with code that also evaluates
                            # a policy and applies a control signal before stepping the physics.
                            mujoco.mj_step(m, d)

                            # Pick up changes to the physics state, apply perturbations, update options from GUI.
                            viewer.sync()
                            
                            # Save data
                            # if not check_graph_collisions(d, graph):
                            
                            # Sample a trajectory given the pose
                            sampler.set_pose_with_traj(graph, traj_counter)
                            
                            pose_dict = sampler.graph_to_pose_dict(graph)
                            set_q(d.qpos, pose_dict)  
                            
                            # Save data
                            if not check_graph_collisions(d, graph):
                                datasaver.append_graph(graph)
                            
                            traj_counter += 1
                            
                            # time.sleep(dt)

            # Save data
            path_dir = os.path.join("data/robot_graphs_trajectories")
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            datasaver.df.to_pickle(os.path.join(path_dir, name + ".pkl"))                   

    return 0


if __name__ == "__main__":
    main()
