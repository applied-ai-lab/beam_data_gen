import torch

from beam_data_gen.traj_opt.dual_assembly import DualAssembly, TrajOptParams, DualArmStates, StateParams


def test():
    
    device = torch.device("cpu")
    
    # State params
    state_params = StateParams(state_dim=5,
                                no_beams=3,
                                no_hands=2,
                                device=device,
                                tol=1.0e-3)
    
    states = DualArmStates(state_params)
    states.beam_goal = torch.ones(states.beam_goal.shape).to(device)
    
    params = TrajOptParams(step_size=0.01,
                            no_steps=100,
                            epsilon=1.e-2,
                            no_particles=30, 
                            device=device)
    
    dual_arm_assembly = DualAssembly(params, 
                                    state_params, 
                                    sim=None,
                                    left_start=None,
                                    right_start=None,
                                    model=None,
                                    data=None)
    
    dual_arm_assembly.states = states
    dual_arm_assembly.states.advance()
    
    gradients = dual_arm_assembly.gradients()
    
    print(f" Beam gradients: \n{gradients.beam_poses} ")
    print(f" Pre-grasp gradients: \n{gradients.pregrasp} ")
    print(f" Left gradients: \n{gradients.left_pose} ")
    print(f" Right gradients: \n{gradients.right_pose} ")
    
    return


if __name__ == "__main__":
    test()
