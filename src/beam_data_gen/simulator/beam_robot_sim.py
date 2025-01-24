from typing import List

import numpy as np

from beam_data_gen.beam_impl.robot_graph import RobotGraph


class BeamRobotSim:
    def __init__(self):
        pass
    
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