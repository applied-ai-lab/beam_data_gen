import copy

import numpy as np

from assembly_tools.graph_primitives.graph_primitives import SquareGraph
from assembly_tools.ramp_graph import RampGraph
from assembly_tools.types import BeamTypeEnum, PoseType, R


class SquareParams(SquareGraph):
    def __init__(self, assembly_id="square"):
        super().__init__(assembly_id)
        
    @property
    def A(self) -> np.array:
        if self._A is None:
            # A upper triangular
            self._A = np.array([
                [0, 1, 1, 0, 0, 0, 1, 1], # Beam 1 
                [1, 0, 1, 0, 0, 0, 0, 0], # Pin  A
                [1, 1, 0, 1, 1, 0, 0, 0], # Beam 2
                [0, 0, 1, 0, 1, 0, 0, 0], # Pin  B
                [0, 0, 1, 1, 0, 1, 1, 0], # Beam 3
                [0, 0, 0, 0, 1, 0, 1, 0], # Pin  C
                [1, 0, 0, 0, 1, 1, 0, 1], # Beam 4
                [1, 0, 0, 0, 0, 0, 1, 0]  # Pin  D
            ]).astype(int)
        return self._A
        
    @property
    def node_dict(self) -> np.array:
        if self._node_dict is None:
            self._node_dict = {
                self._id + "_beam_1": {'type': BeamTypeEnum.BEAM, 
                                       'pose': PoseType(trans=np.array([0.000, 0.000, 0.04]), orient=R.from_quat([0, 0, 0, 1])),
                                       '_l_p': PoseType(trans=np.array([0.000, 0.000, 0.04]), orient=R.from_quat([0, 0, 0, 1])) # Local pose
                                       },
                self._id + "_pin_A":  {'type': BeamTypeEnum.PIN , 
                                       'pose': PoseType(trans=np.array([0.316, 0.000, 0.08]), orient=R.from_quat([0, 0, 0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([0.316, 0.000, 0.08]), orient=R.from_quat([0, 0, 0.707, 0.707])) # Local pose
                                       },
                self._id + "_beam_2": {'type': BeamTypeEnum.BEAM, 
                                       'pose': PoseType(trans=np.array([0.316, 0.277, 0.04]), orient=R.from_quat([0, 0, 0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([0.316, 0.277, 0.04]), orient=R.from_quat([0, 0, 0.707, 0.707])) # Local pose
                                       },
                self._id + "_pin_B":  {'type': BeamTypeEnum.PIN , 
                                       'pose': PoseType(trans=np.array([0.316, 0.554, 0.08]), orient=R.from_quat([0, 0, -0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([0.316, 0.554, 0.08]), orient=R.from_quat([0, 0, -0.707, 0.707])) # Local pose
                                       },
                self._id + "_beam_3": {'type': BeamTypeEnum.BEAM, 
                                       'pose': PoseType(trans=np.array([0.000, 0.554, 0.04]), orient=R.from_quat([0, 0, 1, 0])),
                                       '_l_p': PoseType(trans=np.array([0.000, -0.554, 0.04]), orient=R.from_quat([0, 0, 1, 0])) # Local pose
                                       },
                self._id + "_pin_C":  {'type': BeamTypeEnum.PIN , 
                                       'pose': PoseType(trans=np.array([-0.316, 0.554, 0.08]), orient=R.from_quat([0, 0, -0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([-0.316, 0.554, 0.08]), orient=R.from_quat([0, 0, -0.707, 0.707])) # Local pose
                                       },
                self._id + "_beam_4": {'type': BeamTypeEnum.BEAM, 
                                       'pose': PoseType(trans=np.array([-0.316, 0.277, 0.04]), orient=R.from_quat([0, 0, -0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([-0.316, 0.277, 0.04]), orient=R.from_quat([0, 0, -0.707, 0.707])) # Local pose
                                       },
                self._id + "_pin_D":  {'type': BeamTypeEnum.PIN , 
                                       'pose': PoseType(trans=np.array([-0.316, 0.000, 0.08]), orient=R.from_quat([0, 0, 0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([-0.316, 0.000, 0.08]), orient=R.from_quat([0, 0, 0.707, 0.707])) # Local pose
                                       },
                
            }
        return self._node_dict
    
# Implementations
beam_params = SquareParams()

# Fully connected
square_connected_graph = RampGraph().create_graph(beam_params.A, beam_params.node_dict)
