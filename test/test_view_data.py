import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from beam_data_gen.models.beam_dataset import BeamDataset, ProcessData
from beam_data_gen.simulator.beam_robot_sim import BeamRobotSim


def main():
    m = mujoco.MjModel.from_xml_path('resources/configs/robot_and_beams.xml')
    d = mujoco.MjData(m)
    
    # Beam robot sim
    sim = BeamRobotSim()
    
    # Process data
    process_data = ProcessData(np.array([0.6, 0.6, 0.08]), ["robot_left_hand", "robot_right_hand", "l_beam_1", "l_beam_2", "l_pin_A"])
    poses, flat_adj = process_data("data/robot_graphs_small/")

    poses = torch.tensor(poses, dtype=torch.float32, device=torch.device("cuda"))
    # Create dataset and dataloaders
    denorm_out = process_data.denorm_output(poses)
    
    beams_out = denorm_out[:, 2*5:]
    robot_out = denorm_out[:, 0:2*5]
    
    k = 0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Start loop and sample pose
        while viewer.is_running():
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            
            # Set position
            d.qpos[0:3] = beams_out[k, 0:3].cpu().numpy()
            d.qpos[7:10] = beams_out[k, 5:8].cpu().numpy()
            d.qpos[14:17] = beams_out[k, 10:13].cpu().numpy()
            
            d.qpos[21:24] = robot_out[k, 0:3].cpu().numpy()
            d.qpos[28:31] = robot_out[k, 5:8].cpu().numpy()
            
            # Set orientation
            l1_z = R.from_euler("xyz", [0, 0, beams_out[k, 3].cpu().numpy()])
            l2_z = R.from_euler("xyz", [0, 0, beams_out[k, 8].cpu().numpy()])
            pa_z = R.from_euler("xyz", [0, 0, beams_out[k, 13].cpu().numpy()])
            
            robot_left = R.from_euler("xyz", [0, 0, robot_out[k, 3].cpu().numpy()])
            robot_right = R.from_euler("xyz", [0, 0, robot_out[k, 8].cpu().numpy()])
            
            d.qpos[4:7]   = l1_z.as_quat()[0:3]
            d.qpos[3]   = l1_z.as_quat()[3]
            d.qpos[11:14] = l2_z.as_quat()[0:3]
            d.qpos[10] = l2_z.as_quat()[3]
            d.qpos[18:21] = pa_z.as_quat()[0:3]
            d.qpos[17] = pa_z.as_quat()[3]    
            
            d.qpos[25:28] = robot_left.as_quat()[0:3]
            d.qpos[24] = robot_left.as_quat()[3]
                    
            d.qpos[32:35] = robot_right.as_quat()[0:3]
            d.qpos[31] = robot_right.as_quat()[3]
            
            print(flat_adj[k, :].reshape(5, 5))
            
            import pdb
            pdb.set_trace()
            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            
            k += 1
            if k > poses.shape[0]:
                k = 0
    
            import pdb; pdb.set_trace()   


    return 0


if __name__ == "__main__":
    main()

