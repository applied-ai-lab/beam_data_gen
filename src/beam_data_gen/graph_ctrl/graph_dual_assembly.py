import numpy as np
import scipy.sparse as sp

from beam_data_gen.graph_ctrl.controllers import PickPlaceWithPregrasp
from beam_data_gen.graph_ctrl.graph_dual_assembly import CWiseControllers 


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
        self._x_comp_mat = sp.zeros([self._params.no_components, self._state_dim])
        self._x_tar_mat = sp.zeros([self._params.no_components, self._state_dim])
        self._x_hand_mat = sp.zeros([self._params.no_hands, self._state_dim])

        # Left and right index
        self._left_index = None
        self._right_index = None
        
    def intialise(self, x_c_mat, x_tar_mat):
        self.x_c_mat = x_c_mat
        self.x_c_tar = x_c_tar
        
        self.controllers.initialise(x_c_mat, x_c_tar)
        return
        
    def advance(self, x_hands, x_comps):
        self._x_hand_mat = x_hands
        self._x_comp_mat = x_comps
        
        # Check for convergence
        if self.check_convergence():
            return x_hands, x_comps
        
        else:
           x_left = self._controllers.advance(self._keys[self._left_index],
                                                self._x_hand_mat[0, :], 
                                                self._x_comp_mat[self._left_index, :], 
                                                self._x_tar_mat[self._left_index, :]
                                                )
           x_right = self._controllers.advance(self._keys[self._right_index]
                                                self._x_hand_mat[0, :], 
                                                self._x_comp_mat[self._right_index, :], 
                                                self._x_tar_mat[self._right_index, :]
                                                )
           return x_left, x_right
        
    def check_convergence(self):
        # Calculate the target loss of each component
        self._tar_loss = self.row_wise_loss(self._x_comp_mat, self._x_tar_mat)
        
        # Indices        
        left_dict = {}
        right_dict = {}
        
        for k in range(self._params.no_components):
            if self._tar_loss[k] > self._params.conv_tol:
                # Left
                self._x_comp_mat[k, 1] > 0.0:
                    left_dict[k] = self._tar_loss[k]
                    
                # Right
                self._x_comp_mat[k, 1] > 0.0:
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
        
