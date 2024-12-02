import time

import numpy as np
import mujoco
from mujoco import MjModel, MjData
import mujoco.viewer

from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected)
from beam_data_gen.beam_sampler import BeamSampler

# Function to check for collisions
def check_collisions(data):
    """
    Checks for collisions in the MuJoCo simulation.
    Returns True if any pair of geoms are in contact.
    """
    for i in range(data.ncon):  # Iterate through contacts
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        print(f"Collision detected between geom {geom1} and geom {geom2}")
        return True  # Collision detected
    return False  # No collision


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


def set_x(x_pos, q_pos_dict):
    x_pos[1, :] = q_pos_dict["l_beam_1"].trans
    x_pos[2, :] = q_pos_dict["l_beam_2"].trans
    x_pos[3, :] = q_pos_dict["l_pin_A"].trans


def main():

    m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
    d = mujoco.MjData(m)

    # Initialise the classes
    trans_lims = [0.3, 0.3, 0.0]
    sampler = BeamSampler(trans_lims)

    # Beam config graph
    graph = l_connected_graph

    # Start loop and sample pose
    with mujoco.viewer.launch_passive(m, d) as viewer:
      while viewer.is_running():
        step_start = time.time()
        
        # d.qpos[7] += 0.5 * np.random.randn(1)   
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Check for collisions
        if check_collisions(d):
            print("Cuboids are in collision.")

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        
        import pdb; pdb.set_trace()
        
        # Resample a pose
        sampler.sample_poses(graph, sampler.uniform_pose_sampler)
        pose_dict = sampler.graph_to_pose_dict(graph)
        set_q(d.qpos, pose_dict)  
        
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
          
    return 0


if __name__ == "__main__":
    main()
