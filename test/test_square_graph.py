import copy
from collections import deque

import numpy as np
from itertools import combinations
import networkx as nx

from beam_data_gen.beam_impl.Square_graph import RampGraph
from beam_data_gen.beam_impl.Square_graph import square_connected_graph
from beam_data_gen.beam_impl.robot_graph import square_robot


def find_graphs(graph: RampGraph):
    graph_dict = {}
    queue = deque()
    queue.append(graph)
    
    def _find_graphs(graph_1: RampGraph):
        candidates = graph_1.find_removal_candidates()
        
        import pdb
        pdb.set_trace()
        
        additions = []
        for node in candidates:            
            new_graph = copy.deepcopy(graph_1)
            new_graph.remove_all_edges_from_node(node)
            
            if len(queue) == 0:
                queue.append(new_graph)
            
            else:
                for _graph in queue:
                    if (_graph.A.astype(int) == new_graph.A.astype(int)).all():
                        return
                    else:
                        additions.append(new_graph)
                queue.extend(additions)        
        return
    
    _find_graphs(graph)
    
    counter = 0
    while len(queue) > 1:
        new_graph = queue.popleft()
        graph_dict[counter] = new_graph
        
        _find_graphs(graph)
        
        print(len(queue))
        
        counter += 1
        
    return graph_dict
            
def has_duplicate_arrays(values: list):
    return any(np.array_equal(a, b) for a, b in combinations(values, 2))       
    
    


def main():
    
    full_graph = copy.deepcopy(square_robot)
    
    graph_dict = full_graph.find_intermediate_graphs(full_graph, 100)    
    
    import pdb
    pdb.set_trace()    
    
    return 0


if __name__ == "__main__":
    main()
