from assembly_tools.utils import (plot_Adj, plot_graph, plt)

from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected)
from beam_data_gen.beam_sampler import BeamSampler


def print_node_data(node_data_lst):
    for (node, data) in node_data_lst:
        print(str(node) + str(data["pose"].trans) + str(data["pose"].orient.as_euler(seq="xyz")))


def main():

    trans_lims = [0.3, 0.3, 0.0]
    sampler = BeamSampler(trans_lims)
    
    print_node_data(l_connected_graph.node_lst)
    
    sampler.sample_poses(l_connected_graph, sampler.uniform_pose_sampler)   
    
    print_node_data(l_connected_graph.node_lst) 

    plot_graph(l_connected_graph.graph)
    plot_graph(l_pin_removed.graph)
    plot_graph(l_disconnected.graph)
    plt.show()    
    
    return 0


if __name__ == "__main__":
    main()
