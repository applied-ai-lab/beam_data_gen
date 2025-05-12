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
        # components = [G.subgraph(c) for c in nx.connected_components(G)]
        components = nx.connected_components(G)
        # Iterate through 
        # for sub_graphs in components:
        for node_lst in components:
            # node_lst = list(sub_graphs.nodes(data=True))
            trans, rot = samp_func()
            for node in node_lst:
                data = G.nodes[node]               
                # Sample a transform
                # data["pose"].trans += trans + np.matmul(np.matmul(rot.as_matrix(), data["_l_p"].orient.as_matrix()), data["_l_p"].trans)
                data["pose"].orient = R.from_matrix(np.matmul(rot.as_matrix(), data["_l_p"].orient.as_matrix()))
                data["pose"].trans = trans + np.matmul(data["pose"].orient.as_matrix(), 
                                                        np.matmul(data["_l_p"].orient.as_matrix(), data["_l_p"].trans))
            # Check the graph is feasible
        return ramp_graph.check_graph()
    
    def move_hands(self, ramp_graph: RampGraph, params: PoseSamplerParams):
        # Perturb graph
        self.sample_poses(ramp_graph, self.uniform_pose_sampler)
        # Get the graph
        G = ramp_graph.graph
        # Find the subgraphs
        sub_graphs = [G.subgraph(c) for c in nx.connected_components(G)]
        # Iterate through 
        for sub_graph in sub_graphs:
            # Get types in subgraphs
            type_lst = ramp_graph.get_types_lst(sub_graph)
            
            node_lst = list(sub_graph.nodes(data=True))
            for (node, data) in node_lst:
                    data["traj"] = [copy.deepcopy(data["pose"])] * params.no_samples
            
            if BeamTypeEnum.HAND in type_lst:
                
                # Update params so that beams don't move vertically
                if BeamTypeEnum.BEAM in type_lst:
                    params.velocity_mask[2] = 0.0
                else:
                    params.velocity_mask[2] = 1.0
                    
                # Generate a trajectory
                trans, rot = self.uniform_pose_sampler()
                trans[2] *= 0
                
                trajectory = self.pose_sampler.generate_smooth_transforms(np.hstack([trans, rot.as_quat()]), params)
                
                for (node, data) in node_lst:                    
                    for k in range(trajectory.shape[0] - 1):
                        
                        trans, rot = (trajectory[k, 0:3], 
                                        R.from_quat(trajectory[k, 3:7]))
                        
                        # Sample a transform
                        pose = PoseType(copy.deepcopy(data["traj"][k].trans), copy.deepcopy(data["traj"][k].orient))
                        
                        pose.orient = R.from_matrix(np.matmul(rot.as_matrix(), data["_l_p"].orient.as_matrix()))
                        pose.trans = trans + np.matmul(pose.orient.as_matrix(), 
                                                            np.matmul(data["_l_p"].orient.as_matrix(), data["_l_p"].trans))
                        
                        data["traj"][k] = pose             
                        
        # Check the graph is feasible
        return ramp_graph.check_graph()
    
    def apply_transformations(self, start_position, start_quaternion, transformations):
        """
        Applies a sequence of transformations (position, quaternion) to a start frame.
        
        :param start_position: (3,) array-like, initial position
        :param start_quaternion: (4,) array-like, initial quaternion (x, y, z, w)
        :param transformations: (N, 7) array-like, set of transforms
        
        :return: Final position and quaternion after applying all transformations
        """
        current_position = np.array(start_position)
        current_rotation = R.from_quat(start_quaternion)  # Convert quaternion to rotation object

        N = transformations.shape[0]
        
        new_traj_pos = np.zeros(N, 3)
        new_traj_quat = np.zeros(N, 4)
        
        for k in range(N):
            pos, quat = transformations[k, 0:3], transformations[k, 3:7]
            # Convert the next quaternion to a rotation object
            new_rotation = R.from_quat(quat)
            
            # Apply the rotation to the translation vector
            rotated_translation = current_rotation.apply(pos)
            
            # Update position
            current_position += rotated_translation
            
            # Update orientation
            current_rotation *= new_rotation

        return current_position, current_rotation.as_quat()
    
    def set_pose_with_traj(self, ramp_graph: RampGraph, counter: int):
        node_lst = list(ramp_graph.graph.nodes(data=True))
        for (node, data) in node_lst:
            data["pose"] = data["traj"][counter]
        return ramp_graph.check_graph()        
    
    def graph_to_pose_dict(self, ramp_graph: RampGraph):
        self._node_pose_dict = {}
        nodes_data = ramp_graph.node_lst
        for (node, data) in nodes_data:
            self._node_pose_dict[node] = data["pose"]
        return self._node_pose_dict
            
            
