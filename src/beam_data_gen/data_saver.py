import copy

import pandas as pd 
import numpy as np

from assembly_tools.ramp_graph import RampGraph 


class DataSaver:
    def __init__(self, ramp_graph: RampGraph):
        self._node_names = list(ramp_graph.graph.nodes())
        
        self._column_names = copy.deepcopy(self._node_names) + ["adj_mat"]
        
        self._df = pd.DataFrame(columns=self._column_names)
        
        self._pose = np.zeros(7) # pose [xyz, q_xyzw]
        
    def append_graph(self, ramp_graph: RampGraph):
        data_lst = []
        for node_name in self._node_names:
            pose_data = ramp_graph.graph.nodes[node_name]["pose"]
            self._pose[0:3] = pose_data.trans
            self._pose[3:] = pose_data.orient.as_quat()
            
            data_lst.append(copy.deepcopy(self._pose))
        
        # Append the 
        data_lst.append(ramp_graph.A.flatten())
        
        self._df.loc[len(self._df)] = data_lst
        return
    
    @property
    def df(self):
        return self._df
        
    
    
    

