import torch

from vae_planner.models.container_base import VaeInputsBase, LatentVarsBase, VaeOutputsBase


class BeamRobotInputs(VaeInputsBase):
    def __init__(self):
        super().__init__()
        # self.robot = VaeInputsBase()
        # self.beams = VaeInputsBase()    
        

class BeamRobotLatents(LatentVarsBase):
    def __init__(self):
        super().__init__()
        self.robot = LatentVarsBase()
        self.beams = LatentVarsBase()
        
    @property
    def mu(self):
        self._mu = torch.cat([self.robot.mu, self.beams.mu], 1)
        return self._mu
    
    @property
    def log_var(self):
        self._log_var = torch.cat([self.robot.log_var, self.beams.log_var], 1)
        return self._log_var
    
    @property
    def z(self):
        self._z = torch.cat([self.robot.z, self.beams.z], 1)
        return self._z
        

class BeamRobotOutputs(VaeOutputsBase):
    def __init__(self):
        super().__init__()
        self.robot = VaeOutputsBase()
        self.beams = VaeOutputsBase()
