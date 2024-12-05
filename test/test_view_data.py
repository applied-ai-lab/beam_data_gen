import mujoco
import mujoco.viewer
import numpy as np
import torch

from beam_data_gen.models.beam_dataset import BeamDataset, ProcessData


def main():
    m = mujoco.MjModel.from_xml_path('resources/configs/three_beams.xml')
    d = mujoco.MjData(m)
    
    # Process data
    process_data = ProcessData(np.array([0.6, 0.6, 0.08]))
    poses, flat_adj = process_data("data/graphs/", ["l_beam_1", "l_beam_2", "l_pin_A"])

    poses = torch.tensor(poses, dtype=torch.float32, device=torch.device("cuda"))
    # Create dataset and dataloaders
    denorm_out = process_data.denorm_output(poses)
    
    k = 0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Start loop and sample pose
        while viewer.is_running():
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            
            d.qpos[0:3] = denorm_out[k, 0:3].cpu().numpy()
            d.qpos[7:10] = denorm_out[k, 5:8].cpu().numpy()
            d.qpos[14:17] = denorm_out[k, 10:13].cpu().numpy()
            
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

