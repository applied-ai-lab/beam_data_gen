import numpy as np

from beam_data_gen.traj_opt.p_controllers import PickPlaceWithPregrasp, PseudoProbs


def main():
    
    # Initialise PickPlaceCtrl
    state_dim = 5
    p_ctrl = PickPlaceWithPregrasp(state_dim, Ts=0.1)
    
    # Pseudo-probability func hand is in contact
    def contact_p(x_h, x_c, epsilon_c=0.001):
        return float(np.sum((x_h - x_c) ** 2.0) < epsilon_c) 
    
    # Set some test locations
    x_hand = np.array([0., 0., 0.1, np.sin(0.), np.cos(0.)])
    x_comp = np.array([0.1, 0.1, 0.0, np.sin(0.5), np.cos(0.5)])
    x_tar = np.array([0., 0., 0.0, np.sin(0.), np.cos(0.)])
    
    # Set init condition
    probs = PseudoProbs()
    p_ctrl.initialise(x_hand, x_comp, x_tar)
    
    # Forward predict a few steps
    k = 0
    while k < 100:
        # Pseudo-probability hand is in contact        
        x_1 = p_ctrl.advance(x_hand, x_comp, x_tar)
        
        probs = p_ctrl._probs
        
        x_hand = x_1[0: state_dim, :] 
        x_comp = x_1[4 * state_dim: 5 * state_dim, :]
        
        if k % 1 == 0:
            print(f" p_d: {probs.p_d}, \
                Hand: {x_hand[0:3, :].reshape(1, -1)} \
                and Pregrasp: {x_1[10:13, 0].reshape(1, -1)}")
            print(f" p_C: {probs.p_c}, \
                Hand: {x_hand[0:3, :].reshape(1, -1)}, \
                and Comp: {x_comp[0:3, 0].reshape(1, -1)}")
            
        # if probs.p_c > 0.5 and probs.p_d > 0.5:
        #     import pdb
        #     pdb.set_trace()
        
        k += 1
    
    return 0


if __name__ == "__main__":
    main()
