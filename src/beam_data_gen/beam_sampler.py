import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as R

from assembly_tools.ramp_graph import RampGraph


class BeamSampler:
    def __init__(self, trans_lims: np.array):
        self._trans_lims = trans_lims
        # Pose dict 
        self._node_pose_dict = None
    
    def uniform_pose_sampler(self):
        trans = np.array(list(np.random.uniform(-self._trans_lims[k], self._trans_lims[k]) for k in range(3)))
        # Do not sample roll and pitch
        yaw = np.random.uniform(0, 2 * np.pi)
        rot = R.from_euler(seq="xyz", angles=[0, 0, yaw])
        return trans, rot        
    
    def sample_poses(self, ramp_graph: RampGraph, samp_func):
        # Get the graph
        G = ramp_graph.graph
        # Find the subgraphs
        sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
        # Iterate through 
        for sub_graphs in sub_graphs:
            node_lst = list(sub_graphs.nodes(data=True))
            trans, rot = samp_func()
            for (node, data) in node_lst:
                # Sample a transform
                data["pose"].trans += trans
                data["pose"].orient = R.from_matrix(np.matmul(rot.as_matrix(), data["pose"].orient.as_matrix())) 
        return
    
    def graph_to_joint_angles(self, ramp_graph: RampGraph):
        nodes_data = ramp_graph.node_lst
        for (node, data) in nodes_data:
            self._node_pose_dict[node] = data["pose"]
        return self._node_pose_dict
            
            
