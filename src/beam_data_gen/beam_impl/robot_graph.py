import copy

import numpy as np

from assembly_tools.graph_primitives.graph_primitives import BeamBase
from assembly_tools.ramp_graph import RampGraph
from assembly_tools.types import BeamTypeEnum, PoseType, R

from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected)


class RobotGraphParams(BeamBase):
    def __init__(self, assembly_id="robot"):
        super().__init__(assembly_id)
        self._A = np.zeros([2, 2])
        
    @property
    def node_dict(self) -> np.array:
        if self._node_dict is None:
            self._node_dict = {
                self._id + "_left_hand": {'type': BeamTypeEnum.HAND, 
                                       'pose': PoseType(trans=np.array([0.316, 0.277, 0.25]), orient=R.from_quat([0, 0, 0, 1])),
                                       '_l_p': PoseType(trans=np.array([0.316, 0.277, 0.25]), orient=R.from_quat([0, 0, 0, 1])) # Local pose
                                       },
                self._id + "_right_hand": {'type': BeamTypeEnum.HAND, 
                                       'pose': PoseType(trans=np.array([0.316, -0.277, 0.25]), orient=R.from_quat([0, 0, 0, 1])),
                                       '_l_p': PoseType(trans=np.array([0.316, -0.277, 0.25]), orient=R.from_quat([0, 0, 0, 1])) # Local pose
                                       }
            }
        return self._node_dict
    

class RobotGraph(RampGraph):
    def __init__(self):
        super().__init__()
        
    # Check that the links in the graph match with the object poses
    def check_graph(self):
        # Get nodes
        node_list = list(self._graph.nodes(data=True))
        # Iterate through and check if any are nodes
        for (node, data) in node_list:
            if data['type'] == BeamTypeEnum.HAND:
                # Check the degree is one or fewer
                assert self._graph.degree[node] <= 1, "The hand is grasping too many objects"
                # Check pose of the hand connected to the object
                if self._graph.degree[node] == 1:
                    beam_nodes = list(self._graph.adj[node])
                    data['pose'] = copy.deepcopy(self._graph.nodes[beam_nodes[0]]['pose'])
        return True         
    
    # Add hand
    def add_hand(self, hand_node: str, beam_node: str = None):
        self.remove_edge(hand_node)
        # If beam node is None, remove hand connections
        if beam_node is None:
            self.remove_edge(hand_node)
            return self.check_graph()
        else:
            return self.add_edge(hand_node, beam_node)
    
    # Add edge to graph
    def add_edge(self, node_0: str, node_1: str):
        
        assert node_0 in self.graph.nodes, "node_0 not in graph"
        assert node_1 in self.graph.nodes, "node_1 not in graph"
        
        self.graph.add_edge(node_0, node_1)
        return self.check_graph()
    
    
    # Remove edges from node    
    def remove_edge(self, node: str):
        child_nodes = list(self.graph.adj[node])
        for child_node in child_nodes:
            self.graph.remove_edge(node, child_node)
        return        

    
# Implementations

# Fully connected
l_connected_robot = RobotGraph()
robot_hand_params = RobotGraphParams()
l_connected_robot.create_graph(copy.deepcopy(robot_hand_params.A), 
                               copy.deepcopy(robot_hand_params.node_dict))

l_connected_robot.append_graph(copy.deepcopy(l_connected_graph))

# Pin removed
l_pin_removed_robot = RobotGraph()
l_pin_removed_robot.create_graph(copy.deepcopy(robot_hand_params.A), 
                                 copy.deepcopy(robot_hand_params.node_dict))

l_pin_removed_robot.append_graph(copy.deepcopy(l_pin_removed))

# Fully disconnected
l_disconnected_robot = RobotGraph()
l_disconnected_robot.create_graph(copy.deepcopy(robot_hand_params.A),
                                  copy.deepcopy(robot_hand_params.node_dict))

l_disconnected_robot.append_graph(copy.deepcopy(l_disconnected))
