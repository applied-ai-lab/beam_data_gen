import copy

import numpy as np

from assembly_tools.graph_primitives.graph_primitives import LGraph
from assembly_tools.ramp_graph import RampGraph
from assembly_tools.types import BeamTypeEnum, PoseType, R


class L_BeamParams(LGraph):
    def __init__(self, assembly_id="l"):
        super().__init__(assembly_id)
        
    @property
    def node_dict(self) -> np.array:
        if self._node_dict is None:
            self._node_dict = {
                self._id + "_beam_1": {'type': BeamTypeEnum.BEAM, 
                                       'pose': PoseType(trans=np.array([0.000, 0.000, 0.04]), orient=R.from_quat([0, 0, 0, 1])),
                                       '_l_p': PoseType(trans=np.array([0.000, 0.000, 0.04]), orient=R.from_quat([0, 0, 0, 1])) # Local pose
                                       },
                self._id + "_beam_2": {'type': BeamTypeEnum.BEAM, 
                                       'pose': PoseType(trans=np.array([0.316, 0.277, 0.04]), orient=R.from_quat([0, 0, 0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([0.316, 0.277, 0.04]), orient=R.from_quat([0, 0, 0.707, 0.707])) # Local pose
                                       },
                self._id + "_pin_A":  {'type': BeamTypeEnum.PIN , 
                                       'pose': PoseType(trans=np.array([0.316, 0.000, 0.08]), orient=R.from_quat([0, 0, 0.707, 0.707])),
                                       '_l_p': PoseType(trans=np.array([0.316, 0.000, 0.08]), orient=R.from_quat([0, 0, 0.707, 0.707])) # Local pose
                                       },
            }
        return self._node_dict
    
# Implementations

# Fully connected
l_connected_graph = RampGraph()
beam_params = L_BeamParams()
l_connected_graph.create_graph(beam_params.A, beam_params.node_dict)

# Fully disconnected
l_disconnected = copy.deepcopy(l_connected_graph)

# Pin removed
action_lst = l_disconnected.disassemble()
l_pin_removed = RampGraph()
l_pin_removed.graph = action_lst[0].graph
