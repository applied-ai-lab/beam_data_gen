import numpy as np
import scipy.sparse as sp
from typing import List

from beam_data_gen.graph_ctrl.controllers import PickPlaceWithPregrasp
from beam_data_gen.graph_ctrl.cwise_controllers import CWiseControllers 


class AssemblyParams:
    def __init__(self, 
                 state_dim:int,
                 no_hands:int, 
                 no_components:int,
                 conv_tol: float):
        
        self.state_dim = state_dim
        self.no_hands = no_hands
        self.no_components = no_components
        self.conv_tol = conv_tol


class GraphDualAssembly:
    def __init__(self, params: AssemblyParams, ctrl_type: PickPlaceWithPregrasp, comp_names: List):
        
        self._params = params
        self._state_dim = params.state_dim
        self._keys = comp_names
        self._controllers = CWiseControllers(self._state_dim,
                                             ctrl_type,
                                             comp_names)
        # Losses
        self._tar_loss = 10000.0 * np.ones(self._params.no_components)
        
        # State containers
        self._x_comp_mat = np.zeros([self._params.no_components, self._state_dim])
        self._x_tar_mat = np.zeros([self._params.no_components, self._state_dim])
        self._x_hand_mat = np.zeros([self._params.no_hands, self._state_dim])

        # Left and right index
        self._left_index = None
        self._right_index = None
        
    def initialise(self, x_c_mat, x_tar_mat):
        self._x_comp_mat = x_c_mat
        self._x_tar_mat = x_tar_mat
        
        self._controllers.initialise(x_c_mat, x_tar_mat)
        return
        
    def advance(self, x_hands, x_comps):
        self._x_hand_mat = x_hands
        self._x_comp_mat = x_comps
        
        # Set x_c values
        self._controllers.set_x_c(self._x_comp_mat)
        
        # Calc the pseudo p
        self.p_dict = self._controllers.calc_pseudo_p()
        
        # Check for convergence
        if self.check_convergence(self.p_dict):
            return x_hands, x_comps
        
        else:
            if self._left_index:
                x_left = self._controllers.advance(self._keys[self._left_index],
                                                    self._x_hand_mat[0, :], 
                                                    self._x_comp_mat[self._left_index, :], 
                                                    self._x_tar_mat[self._left_index, :]
                                                    )
                
                x_hands[0, :] = x_left[0: self._state_dim, 0]
                x_comps[self._left_index, :] = x_left[2 * self._state_dim: 3 * self._state_dim, 0]                
                
            if self._right_index:
                x_right = self._controllers.advance(self._keys[self._right_index],
                                                    self._x_hand_mat[1, :], 
                                                    self._x_comp_mat[self._right_index, :], 
                                                    self._x_tar_mat[self._right_index, :]
                                                    )
        
                x_hands[1, :] = x_right[0: self._state_dim, 0]
                x_comps[self._right_index, :] = x_right[2 * self._state_dim: 3 * self._state_dim, 0]
            
            return x_hands, x_comps
       
    def plan(self, x_hands, x_comps, no_iters: int):
        
        # Create init trajectory
        x_traj = np.zeros((no_iters + 1, (self._params.no_hands + self._params.no_components) * self._state_dim))
        
        x_traj[0, 0:self._state_dim] = x_hands[0, :]
        x_traj[0, self._state_dim:2*self._state_dim] = x_hands[1, :]
        x_traj[0, 2*self._state_dim:] = x_comps.reshape(1, -1)
        
        comp_start = self._params.no_hands * self._state_dim
        
        for k in range(no_iters):
            x_hands, x_comps = self.advance(x_hands, x_comps)
            
            # Update traj
            x_traj[k + 1, :] = x_traj[k, :]
            x_traj[k + 1, 0: self._state_dim] = x_hands[0, :]
            x_traj[k + 1, self._state_dim: 2*self._state_dim] = x_hands[1, :]
            
            x_traj[k + 1, comp_start:] = x_comps.reshape(1, -1)
        return x_traj
                
    def check_convergence(self, p_dict):
        # Calculate the target loss of each component
        self._tar_loss = self.row_wise_loss(self._x_comp_mat, self._x_tar_mat)
        
        # Indices        
        left_dict = {}
        right_dict = {}
        
        for k in range(self._params.no_components):                       
            if p_dict[self._keys[k]].p_c_star > 0.5 and p_dict[self._keys[k]].p_b_star > 0.5:
                continue
            else:
                # Left
                if self._x_comp_mat[k, 1] > 0.0:
                    left_dict[k] = self._tar_loss[k]
                    
                # Right
                if self._x_comp_mat[k, 1] < 0.0:
                    right_dict[k] = self._tar_loss[k]
        # Check if no tasks to do -- converged
        if len(left_dict.keys()) == 0 and len(right_dict.keys()) == 0:
            return True
        
        if len(left_dict.keys()) > 0:
        
            # Find the left smallest index
            min_index = min(range(len(left_dict.values())), key=list(left_dict.values()).__getitem__)
            self._left_index = list(left_dict.keys())[min_index]
            
            # Remove the left index from the right arm -- at this point, they are the same
            if self._left_index in right_dict.keys():
                right_dict.pop(self._left_index)
            
        else:            
            self._left_index = None
            
        if len(right_dict.keys()) > 0:
        
            min_index = min(range(len(right_dict.values())), key=list(right_dict.values()).__getitem__)
            self._right_index = list(right_dict.keys())[min_index]
            
        else:            
            self._right_index = None
            
        return False

    @staticmethod
    def row_wise_loss(x, x_tar):
        return np.sum((x - x_tar) ** 2.0, axis=1)
    
    # Setters and Getters
    @property
    def x_comp_mat(self):
        return self._x_comp_mat
    
    @property
    def x_tar_mat(self):
        return self._x_tar_mat
    
    @property
    def x_hand_mat(self):
        return self._x_hand_mat
    
    @x_comp_mat.setter
    def x_comp_mat(self, x):
        self._x_comp_mat = x.reshape(self._params.no_components, self._state_dim).copy()
        return
    
    @x_tar_mat.setter
    def x_tar_mat(self, x):
        self._x_tar_mat = x.reshape(self._params.no_components, self._state_dim).copy()
        return
        
    @x_hand_mat.setter
    def x_hand_mat(self, x):
        self._x_hand_mat = x.reshape(self._params.no_hands, self._state_dim).copy()
        
