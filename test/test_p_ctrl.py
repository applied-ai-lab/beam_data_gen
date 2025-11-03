import numpy as np

from beam_data_gen.traj_opt.p_controllers import PickPlaceCtlr


def main():
    
    # Initialise PickPlaceCtrl
    state_dim = 5
    p_ctrl = PickPlaceCtlr(state_dim)
    
    # Pseudo-probability func hand is in contact
    def contact_p(x_h, x_c, epsilon_c=0.001):
        return float(np.sum((x_h - x_c) ** 2.0) < epsilon_c) 
    
    # Set some test locations
    x_hand = np.array([0., 0., 0.1, np.sin(0.), np.cos(0.)])
    x_comp = np.array([1., 1., 0.0, np.sin(0.5), np.cos(0.5)])
    x_tar = np.array([0., 0., 0.0, np.sin(0.), np.cos(0.)])
    
    # Set the p gain
    p_ctrl.p_gain *= 0.5
    
    # Forward predict a few steps
    k = 0
    while k < 10:
        # Pseudo-probability hand is in contact
        p = contact_p(x_hand, x_comp)
    
        x_1 = p_ctrl.advance(x_hand, x_comp, x_tar, p)
        
        x_hand = x_1[0: state_dim] 
        x_comp = x_1[state_dim:]
        
        print(f" Prob: {p}, Hand: {x_hand.reshape(1, -1)}, and Comp: {x_comp.reshape(1, -1)}")
        
        k += 1
    
    return 0


if __name__ == "__main__":
    main()
