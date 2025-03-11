import time

import numpy as np
import mujoco
import mujoco.viewer
from matplotlib import pyplot as plt

from beam_data_gen.transformations.pose_sampler import PoseSampler
from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot, RobotGraph)
from beam_data_gen.simulator.beam_robot_sim import BeamRobotSim
from beam_data_gen.models.beam_dataset import ProcessData


def main():
    
    sampler = PoseSampler()
    
    num_samples = 100
    time_span = 5
    dt = time_span / num_samples
    # Generate smooth velocities
    t, velocity = sampler.generate_smooth_velocities(num_samples, time_span, 42)
    
    # Get initial poses from the graph
    graph = l_connected_robot
    nodes_and_data = graph.node_lst
    
    # Find beams
    init_poses = [nodes_and_data['l_beam_1']["pose"].to_pose_quat(),
                  nodes_and_data['l_pin_A']["pose"].to_pose_quat(),
                  nodes_and_data['l_beam_2']["pose"].to_pose_quat()]
    
    # Apply velocitiies    
    modified_poses = np.array([
        sampler.apply_velocities_to_poses(init_poses[k], velocity, dt)
        for k in range(len(init_poses))
    ])   
    
    # Visualise the trajectories
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)
    
    
    data_processor = ProcessData(np.array([0.6, 0.6, 0.08]))
    sim = BeamRobotSim(data_processor)
    
    counter = 0
        
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():            
            # Set graph poses using modified poses
            nodes_data = graph.node_lst
            nodes_data['l_beam_1']['pose'].from_pose_quat(modified_poses[0, counter, :])            
            
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
