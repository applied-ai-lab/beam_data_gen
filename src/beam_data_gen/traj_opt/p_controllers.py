import numpy as np


# class PseudoProbs:
#     def __init__(self):
#         self.pregrasp_p
#         self.grasp_p 


class LTIBase:
    def __init__(self, state_dim):
        self._state_dim = state_dim
        
        self._x = np.zeros([self._state_dim, 1])
        self._x1 = np.zeros([self._state_dim, 1])
        self._y1 = np.zeros([self._state_dim, 1])
        
        self._A = np.eye(self._state_dim)
        
        self._B = np.eye(self._state_dim)
        
        self._C = np.eye(self._state_dim)
        
        self._D = np.zeros([self._state_dim, 
                            1])
    
    def advance_lti(self, u):
        
        u = u.reshape(-1, 1)
        
        self._x1 = np.matmul(self._A, self._x) + np.matmul(self._B, u)
        self._y1 = np.matmul(self._C, self._x1) + np.matmul(self._D, u)
        self._x1 = self.x
        return self._y1 
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value.reshape(self._state_dim, 1)
    
    
    
class PickPlaceCtlr(LTIBase):
    def __init__(self, state_dim):
        super().__init__(state_dim)
        
        self._x = np.zeros([2 * self._state_dim, 1])
        self._xdot = np.zeros([2 * self._state_dim, 1])
        
        self.p_gain = np.eye(2 * self._state_dim)
        
        self._A = np.zeros([2 * self._state_dim, 
                            2 * self._state_dim])
        
        self._B = np.zeros([2 * self._state_dim,
                            self._state_dim])
        
        self._C = np.eye(2 * self._state_dim)
        
        self._D = np.zeros([2 * self._state_dim, 
                            self._state_dim])
              
    def advance(self, 
                x_hand: np.array, 
                x_c: np.array, 
                x_c_tar: np.array,
                p: float):
        # Set x value
        self._x[0:self._state_dim, :] = x_hand.reshape(-1, 1)
        self._x[self._state_dim:, :] = x_c.reshape(-1, 1)
        
        u = x_c_tar.reshape(-1, 1)
        
        # Update A, B, etc matrices
        self.A_update(p)
        self.B_update(p)
       
        self._xdot = np.matmul(self._A, self._x) + np.matmul(self._B, u)
        self._x1 = self.x + np.matmul(self.p_gain, self._xdot)
        return self._x1
        
    def A_update(self, p: float):
        # Set to zeros
        self._A *= 0.0
        # Fill the diagonals
        np.fill_diagonal(self._A[0: self._state_dim,
                                    0: self._state_dim], -1.0)
        np.fill_diagonal(self._A[0: self._state_dim,
                                    self._state_dim:], 1. - p)
        np.fill_diagonal(self._A[self._state_dim:,
                                    self._state_dim:], -1.0 * p)
        return
    
    def B_update(self, p: float):
        np.fill_diagonal(self._B[0: self._state_dim,
                                    0: self._state_dim], p)
        np.fill_diagonal(self._B[self._state_dim:,
                                    0: self._state_dim], p)
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value.reshape(2 * self._state_dim, 1)
    
        