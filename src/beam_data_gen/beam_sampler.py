import copy

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as R

from assembly_tools.ramp_graph import RampGraph, BeamTypeEnum
from assembly_tools.types import PoseType

from beam_data_gen.transformations.pose_sampler import PoseSampler, PoseSamplerParams

class BeamSampler:
    def __init__(self, trans_lims: np.array):
        self._trans_lims = trans_lims
        # Pose dict 
        self._node_pose_dict = {}
        # Smooth pose sampler
        self.pose_sampler = PoseSampler()
    
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
                # data["pose"].trans += trans + np.matmul(np.matmul(rot.as_matrix(), data["_l_p"].orient.as_matrix()), data["_l_p"].trans)
                data["pose"].orient = R.from_matrix(np.matmul(rot.as_matrix(), data["_l_p"].orient.as_matrix()))
                data["pose"].trans = trans + np.matmul(data["pose"].orient.as_matrix(), 
                                                       np.matmul(data["_l_p"].orient.as_matrix(), data["_l_p"].trans))
            # Check the graph is feasible
        return ramp_graph.check_graph()
    
    def move_hands(self, ramp_graph: RampGraph, params: PoseSamplerParams):
        # Get the graph
        G = ramp_graph.graph
        # Find the subgraphs
        sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
        # Iterate through 
        for sub_graph in sub_graphs:
            # Get types in subgraphs
            type_lst = ramp_graph.get_types_lst(sub_graph)
            
            if BeamTypeEnum.HAND in type_lst:
                node_lst = list(sub_graph.nodes(data=True))
                    
                # Generate a trajectory
                trajectory = self.pose_sampler.generate_smooth_transforms(params)
                
                for (node, data) in node_lst:
                    data["traj"] = [copy.deepcopy(data["pose"])]
                    
                    for k in range(trajectory.shape[0] - 1):
                        trans, rot = trajectory[k, 0:3], R.from_quat(trajectory[k, 3:7])
                        
                        # Sample a transform
                        pose = PoseType(copy.deepcopy(data["traj"][k].trans), copy.deepcopy(data["traj"][k].orient))
                        
                        pose.orient = R.from_matrix(np.matmul(rot.as_matrix(), data["_l_p"].orient.as_matrix()))
                        pose.trans = trans + np.matmul(pose.orient.as_matrix(), 
                                                            np.matmul(data["_l_p"].orient.as_matrix(), data["_l_p"].trans))
                        
                        data["traj"].append(pose)
                        
        # Check the graph is feasible
        return ramp_graph.check_graph()
    
    def set_pose_with_traj(self, ramp_graph: RampGraph, counter: int):
        node_lst = list(ramp_graph.graph.nodes(data=True))
        for (node, data) in node_lst:
            data["pose"] = data["traj"][counter]
        return ramp_graph.check_graph()        
    
    def graph_to_pose_dict(self, ramp_graph: RampGraph):
        nodes_data = ramp_graph.node_lst
        for (node, data) in nodes_data:
            self._node_pose_dict[node] = data["pose"]
        return self._node_pose_dict
            
            
