import numpy as np
import scipy.sparse as sp
from typing import List

from beam_data_gen.graph_ctrl.controllers import PickPlaceWithPregrasp 


class CWiseControllers:
    def __init__(self, state_dim:int, ctrl_type: PickPlaceWithPregrasp, keys: List):
        
        self._state_dim = state_dim
        self._keys = keys
        self._ctrl_type = ctrl_type
        self._ctrl_dict = self._create_ctrl_dict()
        self._pseudo_p = {}
        
    def initialise(self, x_c, x_c_tar):
        x_hand = np.zeros(self._state_dim)
        for k, (_, item) in enumerate(self._ctrl_dict.items()):
            item.init_x(x_hand, x_c[k, :], x_c_tar[k, :])
        return
    
    def set_x_c(self, x_c):
        for k, (_, item) in enumerate(self._ctrl_dict.items()):
            item.set_x_c(x_c[k, :])
        return
    
    def calc_pseudo_p(self):
        for key, item in self._ctrl_dict.items():
            self._pseudo_p[key] = item.calc_pseudo_p()
        return self._pseudo_p
    
    def advance(self, key, x_hand, x_c, x_c_tar):
        return self._ctrl_dict[key].advance(x_hand, x_c, x_c_tar)
        
    def _create_ctrl_dict(self):
        ctrl_dict = {}
        for key in self._keys:
            ctrl_dict[key] = self._ctrl_type(self._state_dim)
        return ctrl_dict
