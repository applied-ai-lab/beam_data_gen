from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from beam_data_gen.beam_impl.robot_graph import RobotGraph, RampGraph
from beam_data_gen.models.datasets.process_data import ProcessData


class SquareRobotSim:
    def __init__(self, process_data: ProcessData):
        self._data_processor = process_data
        
        self._node_pose_dict = {}
    
    def pose_to_q(self, trans, rot):
        pose = np.zeros(7)
        pose[0:3] = trans
        # Mujoco quat order w x y z
        pose[3] = rot.as_quat()[-1]
        pose[4:] = rot.as_quat()[0:-1]
        return pose 


    def set_q(self, q_pos_dict):
        q_pos = np.zeros(35)
        q_pos[0:7] = self.pose_to_q(q_pos_dict["l_beam_1"].trans, q_pos_dict["l_beam_1"].orient)
        q_pos[7:14] = self.pose_to_q(q_pos_dict["l_beam_2"].trans, q_pos_dict["l_beam_2"].orient)
        q_pos[14:21] = self.pose_to_q(q_pos_dict["l_pin_A"].trans, q_pos_dict["l_pin_A"].orient)
        q_pos[21:28] = self.pose_to_q(q_pos_dict["robot_left_hand"].trans, q_pos_dict["robot_left_hand"].orient)
        q_pos[28:35] = self.pose_to_q(q_pos_dict["robot_right_hand"].trans, q_pos_dict["robot_right_hand"].orient)
        return q_pos
    
    def graph_to_pose_dict(self, ramp_graph: RampGraph):
        nodes_data = ramp_graph.node_lst
        for (node, data) in nodes_data:
            self._node_pose_dict[node] = data["pose"]
        return self._node_pose_dict
        
    def check_graph_collisions(self, data, ramp_graph: RobotGraph):
        collision = False
        geom_to_name = {1: "l_beam_1",
                        2: "l_beam_2",
                        3: "l_pin_A"}
        for i in range(data.ncon):  # Iterate through contacts
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            if geom1 not in geom_to_name.keys() or geom2 not in geom_to_name.keys():
                continue
            # If there is a contact between two items which are not connected return true
            elif not ramp_graph.graph.has_edge(geom_to_name[geom1], geom_to_name[geom2]):
                return True
        return collision
    
    def decode_x(self, data, x_pred):
        denorm_output = self._data_processor.denorm_output(x_pred)[:, :].cpu().detach().numpy()
        beam_out = denorm_output[:, 2*5:]
        robot_out = denorm_output[:, 0:2*5]
        
        pose_len = 7 # Pose includes position (3) and quat (4)
        state_dim = self._data_processor.state_dim
        
        # Update the beams
        no_beams = beam_out.shape[1] // state_dim
        
        def update_data(k, d, x, offset=0):
            # Position
            d.qpos[k*pose_len + offset: k*pose_len + 3 + offset] = x[0, k*state_dim: k*state_dim + 3]
            # Orientation
            q_beam = R.from_euler("xyz", [0, 0, x[0, k*state_dim + 3]])
            # Quat xyz
            d.qpos[k*pose_len + 4 + offset: k*pose_len + 7 + offset]   = q_beam.as_quat()[0:3]
            d.qpos[k*pose_len + 3 + offset]   = q_beam.as_quat()[3]
            return
            
        for k in range(no_beams):
            update_data(k, data, beam_out, offset=0)
            
        # Update the hands
        for k in range(2):
            update_data(k, data, robot_out, offset=no_beams*pose_len)
        return
        