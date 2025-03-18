import copy

from assembly_tools.utils import (plot_Adj, plot_graph, plt)

from beam_data_gen.beam_impl.robot_graph import (l_connected_robot, l_pin_removed_robot, l_disconnected_robot)
from beam_data_gen.data_sampling.beam_sampler import BeamSampler


def print_node_data(node_data_lst):
    for (node, data) in node_data_lst:
        print(str(node) + str(data["pose"].trans) + str(data["pose"].orient.as_euler(seq="xyz")))
        
    
def main():
    
    print(l_connected_robot.A)
    print(l_pin_removed_robot.A)
    print(l_disconnected_robot.A)
    
    l_connected_robot.check_graph()
    
    print(copy.deepcopy(l_connected_robot.graph.nodes['robot_left_hand']['pose'].trans))
    
    l_connected_robot._graph.add_edge("robot_left_hand", "l_beam_1")
    l_connected_robot.check_graph()
    
    print(copy.deepcopy(l_connected_robot.graph.nodes['robot_left_hand']['pose'].trans))
    
    import pdb
    pdb.set_trace()
    
    return 0


if __name__ == "__main__":
    main()
