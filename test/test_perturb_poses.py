import time

import numpy as np
import mujoco
import mujoco.viewer
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from beam_data_gen.transformations.pose_sampler import PoseSampler, PoseSamplerParams
from beam_data_gen.beam_sampler import BeamSampler
from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot, RobotGraph)
from beam_data_gen.simulator.beam_robot_sim import BeamRobotSim
from beam_data_gen.models.beam_dataset import ProcessData


def main():
    
    pose_sampler = PoseSampler()
    beam_sampler = BeamSampler(np.array([0.6, 0.6, 0.08]))
    
    num_samples = 100
    time_span = 5
    dt = time_span / num_samples
    # Generate smooth velocities
    t, velocity = pose_sampler.generate_smooth_velocities(num_samples, time_span, 42)
    velocity *= np.array([1, 1, 0, 0, 0, 1])
    
    params = PoseSamplerParams(num_samples, time_span, None, np.array([1, 1, 0, 0, 0, 1]))
    
    # Get initial poses from the graph
    graph = l_connected_robot
    # Apply hand
    graph.add_hand("robot_left_hand", "l_beam_1")
    # Get nodes and data
    nodes_and_data = graph.node_lst
    
    # Find beams
    init_poses = [nodes_and_data['l_beam_1']["pose"].to_pose_quat(),
                  nodes_and_data['l_pin_A']["pose"].to_pose_quat(),
                  nodes_and_data['l_beam_2']["pose"].to_pose_quat()]
    
    # Apply velocitiies    
    modified_poses = np.array([
        pose_sampler.apply_velocities_to_poses(init_poses[k], velocity, dt)
        for k in range(len(init_poses))
    ])
    
    def sample_func(index_args):
        index = index_args[0]
        trans, rot = modified_poses[0, index, 0:3], R.from_quat(modified_poses[0, index, 3:7])
        index += 1
        index = index % num_samples
        return trans, rot       
        
    # Visualise the trajectories
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)
    
    
    data_processor = ProcessData(np.array([0.6, 0.6, 0.08]))
    sim = BeamRobotSim(data_processor)
    
    beam_sampler.move_hands(graph, params)
    
    counter = 0
        
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and counter < num_samples:            
            # Set pose using the trajectory 
            beam_sampler.set_pose_with_traj(graph, counter)
            
            pose_dict = sim.graph_to_pose_dict(graph)
            d.qpos = sim.set_q(pose_dict)
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            
            time.sleep(0.5)
            
            counter += 1

    
    plt.figure()
    plt.plot(t, velocity[:, 0])
    
    plt.figure()
    plt.plot(t, modified_poses[0, :, 0])
    plt.show()    
    
    return 0


if __name__ == "__main__":
    main()
