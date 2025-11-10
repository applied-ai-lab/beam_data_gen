import numpy as np
import scipy.sparse as sp
import networkx as nx
from matplotlib import pyplot as plt

from beam_data_gen.graph_ctrl.ctrl_graph import CtrlGraph
from beam_data_gen.graph_ctrl.controllers import PickPlaceWithPregrasp 


def main():
    
    # State dim
    state_dim = 5
    
    # Set some test locations
    x_hand = np.array([0., 0., 0.1, np.sin(0.), np.cos(0.)])
    x_comp = np.array([0.1, 0.1, 0.0, np.sin(0.5), np.cos(0.5)])
    x_tar = np.array([0., 0., 0.0, np.sin(0.), np.cos(0.)])
    
    # Controller
    p_ctrl = PickPlaceWithPregrasp(state_dim)
    p_ctrl.init_x(x_hand, x_comp, x_tar)
    
    k = 0
    no_iters = 1000
    
    while k < no_iters:
    
        # Set the x vector
        x1 = p_ctrl.advance(x_hand, x_comp, x_tar)
        
        # # Calc pseudo probs
        # p_tup = p_ctrl.calc_pseudo_p()
        
        # # Hash the pseudo pubs
        # key = p_ctrl.hash_pseudo_p(*p_tup)
        
        # # Get p controller
        # x1 = p_ctrl._trans_dict_p[key](p_ctrl._x)   
        
        # # Update the current state to the next one
        # p_ctrl._x = x1
        
        # Extract data from x
        x_hand = x1[0: state_dim, 0]
        x_comp = x1[2*state_dim:3*state_dim, 0]
        
        print(f' Key: {p_ctrl.key}')
        # print(f' States: \n{x1.reshape(-1, state_dim)}')
        
        k += 1
    
    
    return 0


if __name__ == "__main__":
    main()
