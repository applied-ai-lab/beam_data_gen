import torch

from vae_planner.models.container_base import VaeInputsBase, LatentVarsBase, VaeOutputsBase


class BeamRobotInputs(VaeInputsBase):
    def __init__(self):
        super().__init__()
        # self.robot = VaeInputsBase()
        # self.beams = VaeInputsBase()    
        

class BeamRobotLatents(LatentVarsBase):
    def __init__(self, robot_latent_dim: int=None, beam_latent_dim=None):
        super().__init__()
        self.robot = LatentVarsBase()
        self.beams = LatentVarsBase()
        
        self.robot_latent_dim = robot_latent_dim
        self.beam_latent_dim = beam_latent_dim
        
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
    
    @z.setter
    def z(self, value: torch.tensor):
        if self.robot_latent_dim is not None:
            self.robot.z = value[:, 0:self.robot_latent_dim]
        else:
            raise Exception("Robot latent dim is not set. Please set the robot dim before setting the z values.") 
        if self.beam_latent_dim is not None:
            self.beams.z = value[:, self.robot_latent_dim: self.robot_latent_dim + self.beam_latent_dim]
        else:
            raise Exception("Beam latent dim is not set. Please set the Beam dim before setting the z values.") 
        

class BeamRobotOutputs(VaeOutputsBase):
    def __init__(self):
        super().__init__()
        self.robot = VaeOutputsBase()
        self.beams = VaeOutputsBase()
