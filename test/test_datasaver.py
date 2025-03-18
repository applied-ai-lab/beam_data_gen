from beam_data_gen.data_sampling.data_saver import DataSaver
from beam_data_gen.beam_impl.L_beam import (l_connected_graph, l_pin_removed, l_disconnected)


def test_datasaver():
    
    datasaver = DataSaver(l_connected_graph)
    
    datasaver.append_graph(l_connected_graph)
    
    df = datasaver.df
    
    return 0


if __name__ == "__main__":
    test_datasaver()
