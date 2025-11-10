import numpy as np
import scipy.sparse as sp
import networkx as nx

from beam_data_gen.graph_ctrl.ctrl_graph import CtrlGraph


class PickPlaceWithPregrasp:
    def __init__(self, state_dim):
        
        self._state_dim = state_dim
        
        no_nodes = 5
        
        # State vector
        self._x = np.zeros((self._state_dim * no_nodes, 1))
        
        self._height = 0.08
        
        self.key = "0000"
                 
        # No connections
        A0 = np.zeros([no_nodes, no_nodes])
        
        # Pb* only
        A1 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        
        # Pb and Pb*
        A2 = np.array([[0, 1, 0, 1, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        
        # Pb and Pb*
        A3 = np.array([[0, 1, 0, 0, 0],
                       [1, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        
        # Pb move to Pc
        A4 = np.array([[0, 1, 1, 0, 0],
                       [1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        
        # Pb and Pc
        A5 = np.array([[0, 1, 1, 0, 0],
                       [1, 0, 1, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        
        # Move c to goal
        A6 = np.array([[0, 1, 1, 0, 1],
                       [1, 0, 1, 0, 1],
                       [1, 1, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        
        # C at goal
        A7 = np.array([[0, 1, 1, 0, 1],
                       [1, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [0, 0, 0, 0, 0],
                       [1, 1, 1, 0, 0]])
        
        # Leave C and return to B*
        A8 = np.array([[0, 1, 0, 1, 0],
                       [1, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0]])
        
        # Terminal state at B*
        A9 = np.array([[0, 1, 0, 1, 0],
                       [1, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1],
                       [1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0]])
        
        # Transition dict from pb, pc, pb*, pc*
        self._trans_dict_p = {'0000': CtrlGraph(A2, self._state_dim),
                              '0010': CtrlGraph(A2, self._state_dim),
                              '1010': CtrlGraph(A4, self._state_dim),
                              '1000': CtrlGraph(A4, self._state_dim),
                              '0100': CtrlGraph(A6, self._state_dim),
                              '1100': CtrlGraph(A6, self._state_dim), 
                              '1101': CtrlGraph(A8, self._state_dim),
                              '1001': CtrlGraph(A8, self._state_dim),
                              '1011': CtrlGraph(A0, self._state_dim),
                              '0011': CtrlGraph(A0, self._state_dim)
                              }
        
    def advance(self, x_hand, x_c, x_c_tar):
        # Set the x state
        self.set_x(x_hand, x_c, x_c_tar)
        
        # Calculate the pseudo probs
        p_tup = self.calc_pseudo_p()
        
        # Hash the pseudo probs
        self.key = self.hash_pseudo_p(*p_tup)
        
        # Get the p controller and advance
        self._x = self._trans_dict_p[self.key](self._x)
        return self._x
        
    def hash_pseudo_p(self, pb, pc, pb_star, pc_star):
        return str(int(pb > 0.5)) + str(int(pc > 0.5)) + str(int(pb_star > 0.5)) + str(int(pc_star > 0.5))
    
    @staticmethod
    def contact_p(x_h, x_c, epsilon):
        return float(np.sum((x_h - x_c) ** 2.0) < epsilon)
    
    def calc_pseudo_p(self, epsilon=0.0005):
        p_c = self.contact_p(self._x[0:self._state_dim, 0],
                                  self._x[2 * self._state_dim: 3 * self._state_dim, 0],
                                  epsilon)
        
        p_b = self.contact_p(self._x[0:self._state_dim, 0],
                                  self._x[1 * self._state_dim: 2 * self._state_dim, 0],
                                  epsilon)
        
        p_c_star = self.contact_p(self._x[2 * self._state_dim: 3*self._state_dim, 0],
                                  self._x[4 * self._state_dim: 5 * self._state_dim, 0],
                                  epsilon)
        
        p_b_star = self.contact_p(self._x[self._state_dim: 2 *self._state_dim, 0],
                                  self._x[3 * self._state_dim: 4 * self._state_dim, 0],
                                  epsilon)
        return p_b, p_c, p_b_star, p_c_star
    
    @property
    def x(self):
        return self._x
    
    def init_x(self, x_hand, x_c, x_c_tar):
        self.set_x(x_hand, x_c, x_c_tar)
        self._x[1*self._state_dim: 2*self._state_dim, 0] = self._x[3*self._state_dim: 4*self._state_dim, 0].copy()
        return
    
    def set_x(self, x_hand, x_c, x_c_tar):
        # Update x state
        self._x[0:self._state_dim, 0] = x_hand.copy()
        self._x[2*self._state_dim: 3*self._state_dim, 0] = x_c.copy()
        self._x[3*self._state_dim: 4*self._state_dim, 0] = x_c.copy()
        self._x[3*self._state_dim + 2, 0] += self._height
        self._x[4*self._state_dim: 5*self._state_dim, 0] = x_c_tar.copy()
        return
        
        
