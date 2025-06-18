import numpy as np 
import torch


class TrajOptParams:
    def __init__(self, 
                step_size: float,
                no_steps: int,
                epsilon: float,
                no_particles: int, 
                device: torch.device = torch.device("cuda")):
        
        """ 
        Parameters for trajectory optimisation
        """
        self.step_size = step_size # Step size during optimisation
        self.no_steps = no_steps   # Number of optimisation steps
        self.epsilon = epsilon     # Convergence criteria for optimisation
        self.no_particles = no_particles # Number of particles
        self.device = device # Device
        

class TrajOptBase:
    def __init__(self, params: TrajOptParams):
        self._params = params
        
        self._goal = None
        self._x = None
        
    def optimise(self):
        return        
    
    # Getters and Setters
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, params):
        self._params = params
    
    @property
    def goal(self):
        return self._goal
        
    @goal.setter
    def goal(self, value):
        self._goal = value
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        