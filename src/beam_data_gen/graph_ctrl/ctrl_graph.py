import numpy as np
import scipy.sparse as sp
import networkx as nx


class CtrlGraph:
    def __init__(self, A: np.array, state_dim: int):
        self.A = A
        
        self._state_dim = state_dim
        
        self._D = None
        self._L = None
        
        # State dynamics
        self._Adyn = None
        
        # State vector
        self._x = np.zeros((5 * self._state_dim, 1))
        self._xdot = np.zeros((5 * self._state_dim, 1))
        
        # Gain matrix
        self._gain = 0.05 * sp.eye(self._x.shape[0])
        
        self._height = 0.08
        
    def __call__(self, x):
        return self.advance(x)
        
    def advance(self, x):
        # Update x state
        self._x = x.copy()
        
        self._xdot = self.Adyn @ self._x     
        self._x = self._x + self._gain @ self._xdot
        return self._x   
        
    @property
    def D(self):
        self._D = np.diag(self.A.sum(axis=1))
        return self._D
    
    @property
    def L(self):
        return self.A - self.D
    
    @property
    def Adyn(self):
        self._Adyn = sp.kron(sp.csr_matrix(self.L), np.eye(self._state_dim))
        return self._Adyn
    
    @property
    def gain(self):
        return self._gain
    
    @gain.setter
    def gain(self, gain_diag):
        self._gain = sp.diags(gain_diag)
        return

    def to_graph(self):
        return nx.from_numpy_array(self.A, create_using = nx.DiGraph())
