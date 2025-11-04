import numpy as np


class PseudoProbs:
    def __init__(self):
        self.p_c = 0.0
        self.p_c_star = 0.0
        self.p_d = 0.0
        self.p_d_star = 0.0
        
    def reset(self):
        self.p_c = 0.0
        self.p_c_star = 0.0
        self.p_d = 0.0
        self.p_d_star = 0.0


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
        
  
class PickPlaceWithPregrasp(LTIBase):
    def __init__(self, state_dim: int, Ts: float):
        super().__init__(state_dim)
        
        self._Ts = Ts
        
        self.height = 0.15
        
        self._probs = PseudoProbs()
        
        no_states = 6
        
        self._x = np.zeros([no_states * self._state_dim, 1])
        self._xprime = np.zeros([no_states * self._state_dim, 1])
        
        self._A = np.zeros([no_states * self._state_dim, 
                            no_states * self._state_dim])
        self.A_init()
        
        self._B = np.zeros([no_states * self._state_dim,
                            2 * self._state_dim])
        
        self._C = np.zeros([no_states * self._state_dim, 
                            no_states * self._state_dim])
        self.C_init()
        
        self._u = np.zeros((2 * self._state_dim, 1))
        
    def reset(self):
        self._x *= 0.0
        self._xprime *= 0.0
        self._u *= 0.0
        self._probs.reset()
        return
    
    def initialise(self, 
                   x_hand: np.array, 
                   x_c: np.array,
                   x_c_tar: np.array):
        self.reset()
        # Init the matrices        
        self._x[0:self._state_dim, :] = x_hand.reshape(-1, 1)
        x_d = x_c.reshape(-1, 1).copy()
        x_d[2, 0] += self.height
        self._x[2 * self._state_dim: 3 * self._state_dim, :] = x_d
        self._x[4 * self._state_dim: 5 * self._state_dim, :] = x_c.reshape(-1, 1)   
        
        self._u[0:self._state_dim, :] = x_c.reshape(-1, 1).copy() 
        self._u[2, 0] += self.height
        self._u[self._state_dim: 2*self._state_dim, 0] = x_c_tar
        
        self.probs_update()
        return

    def advance(self, 
                x_hand: np.array, 
                x_c: np.array,
                x_c_tar: np.array):
        # Set x value
        self._x[0:self._state_dim, :] = x_hand.reshape(-1, 1)
        self._x[4 * self._state_dim: 5 * self._state_dim, :] = x_c.reshape(-1, 1)
        
        self._u[0:self._state_dim, :] = x_c.reshape(-1, 1).copy() 
        self._u[2, 0] += self.height
        self._u[self._state_dim: 2*self._state_dim, 0] = x_c_tar
        
        # Update probs
        self.probs_update()
        
        # Update A, B, etc matrices
        self.A_update(self._probs)
        self.B_update(self._probs)
       
        self._xprime = np.matmul(self._A, self._x) + np.matmul(self._B, self._u)
        self._x = self._xprime.copy()
        return self._xprime
    
    def probs_update(self, epsilon=0.001):
        
        self._probs.p_c = self.contact_p(self._x[0:self._state_dim, 0],
                                  self._x[4 * self._state_dim: 5 * self._state_dim, 0],
                                  epsilon)
        
        self._probs.p_d = self.contact_p(self._x[0:self._state_dim, 0],
                                  self._x[2 * self._state_dim: 3 * self._state_dim, 0],
                                  epsilon)
        
        self._probs.p_c_star = self.contact_p(self._u[self._state_dim: 2*self._state_dim, 0],
                                  self._x[4 * self._state_dim: 5 * self._state_dim, 0],
                                  epsilon)
        
        self._probs.p_d_star = self.contact_p(self._u[0:self._state_dim, 0],
                                  self._x[2 * self._state_dim: 3 * self._state_dim, 0],
                                  epsilon)
        return
        
        
    @staticmethod
    def contact_p(x_h, x_c, epsilon):
            return float(np.sum((x_h - x_c) ** 2.0) < epsilon) 
        
        
    def A_update(self, probs: PseudoProbs):
        # h dot update 
        np.fill_diagonal(self._A[self._state_dim: 2 * self._state_dim,
                                    0: self._state_dim], 
                        -(1. - probs.p_c_star) * (1. - probs.p_c))
        np.fill_diagonal(self._A[self._state_dim: 2 * self._state_dim,
                                    2 * self._state_dim: 3 * self._state_dim], 
                        (1. - probs.p_c_star) * (1. - probs.p_c) * (1. - probs.p_d) - probs.p_c_star * (1. - probs.p_d_star))
        np.fill_diagonal(self._A[self._state_dim: 2 * self._state_dim,
                                    4 * self._state_dim: 5 * self._state_dim], 
                        (1. - probs.p_c_star) * (1. - probs.p_c) * (probs.p_d) - (1. - probs.p_c_star) * probs.p_c)
        # # d update
        np.fill_diagonal(self._A[2 * self._state_dim: 3 * self._state_dim,
                                    0: self._state_dim], 
                        probs.p_d) 
        np.fill_diagonal(self._A[2 * self._state_dim: 3 * self._state_dim,
                                    2 * self._state_dim: 3 * self._state_dim], 
                        (1. - probs.p_d))
        np.fill_diagonal(self._A[2 * self._state_dim: 3 * self._state_dim,
                                    3 * self._state_dim: 4 * self._state_dim], 
                        (1. - probs.p_d) * self._Ts)        
        # d dot update
        np.fill_diagonal(self._A[3 * self._state_dim: 4 * self._state_dim,
                                    self._state_dim: 2 * self._state_dim], 
                        probs.p_d)
        np.fill_diagonal(self._A[3 * self._state_dim: 4 * self._state_dim,
                                    2 * self._state_dim: 3 * self._state_dim], 
                        -(1. - probs.p_d))
        # # c dot update
        # np.fill_diagonal(self._A[5 * self._state_dim: 6 * self._state_dim,
        #                             self._state_dim: 2 * self._state_dim], 
        #                 probs.p_c)
        # np.fill_diagonal(self._A[5 * self._state_dim: 6 * self._state_dim,
        #                             4 * self._state_dim: 5 * self._state_dim], 
        #                 (1. - probs.p_c))
        return
    
    def B_update(self, probs: PseudoProbs):
        # h dot inputs
        np.fill_diagonal(self._B[self._state_dim: 2 * self._state_dim,
                                    0: self._state_dim], 
                        probs.p_c_star * (1. - probs.p_d_star))
        np.fill_diagonal(self._B[self._state_dim: 2 * self._state_dim,
                                    self._state_dim: 2 * self._state_dim], 
                        (1. - probs.p_c_star) * probs.p_c)
        # d dot inputs
        np.fill_diagonal(self._B[3 * self._state_dim: 4 * self._state_dim,
                                    0: self._state_dim], 
                        (1. - probs.p_d))
        return
        
    def A_init(self):
        self._A[0: self._state_dim,
                0: self._state_dim] = np.eye(self._state_dim)
        self._A[0: self._state_dim, 
                self._state_dim: 2 * self._state_dim] = self._Ts * np.eye(self._state_dim)
        
        self._A[2 *self._state_dim: 3 * self._state_dim,
                2 *self._state_dim: 3 * self._state_dim] = np.eye(self._state_dim)
        self._A[2 *self._state_dim: 3 * self._state_dim,
                3 *self._state_dim: 4 * self._state_dim] = self._Ts * np.eye(self._state_dim)
        
        self._A[4 *self._state_dim: 5 * self._state_dim,
                4 *self._state_dim: 5 * self._state_dim] = np.eye(self._state_dim)
        self._A[4 *self._state_dim: 5 * self._state_dim,
                5 *self._state_dim: 6 * self._state_dim] = self._Ts * np.eye(self._state_dim)
        return
    
    def C_init(self):
        self._C[0: self._state_dim,
                0: self._state_dim] = np.eye(self._state_dim)
        self._C[2 *self._state_dim: 3 * self._state_dim,
                2 *self._state_dim: 3 * self._state_dim] = np.eye(self._state_dim)
        self._C[4 *self._state_dim: 5 * self._state_dim,
                4 *self._state_dim: 5 * self._state_dim] = np.eye(self._state_dim)
        return
