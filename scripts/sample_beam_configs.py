import time
import os

import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer

from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected, RampGraph)
from beam_data_gen.beam_sampler import BeamSampler
from beam_data_gen.data_saver import DataSaver


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
    return

    
def check_graph_collisions(data, ramp_graph: RampGraph):
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

    m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
    d = mujoco.MjData(m)

    # Initialise the classes
    trans_lims = [0.3, 0.3, 0.0]
    sampler = BeamSampler(trans_lims)

    # Beam config graph
    graphs = {"connected": l_connected_graph, 
              "pin_removed": l_pin_removed, 
              "disconnected": l_disconnected}
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for name, graph in graphs.items():
            # Create the datasaver
            datasaver = DataSaver(graph)

            counter = 0
            max_iters = 1000
            
            # Start loop and sample pose
            while viewer.is_running() and counter < max_iters:
                step_start = time.time()
                
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()
                
                # Save data
                if not check_graph_collisions(d, graph):
                    datasaver.append_to_graph(graph)
                    counter += 1
                
                # Resample a pose
                sampler.sample_poses(graph, sampler.uniform_pose_sampler)
                pose_dict = sampler.graph_to_pose_dict(graph)
                set_q(d.qpos, pose_dict)  
                
                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
            # Save data
            datasaver.df.to_pickle(os.path.join("data/graphs", name + ".pkl"))           
          
    return 0


if __name__ == "__main__":
    main()
