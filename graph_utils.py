import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import simbench as sb

def get_dist_grid_codes(scenario=0):
    # Create the codes for the distribution grid cases of Simbench (LV and MV and any combination of the two)
    codes = sb.collect_all_simbench_codes(scenario=scenario)
    dist_grid_codes = list(filter(lambda x: "no_sw" in x and ("-MV-" in x or "-LV-" in x), codes))
    return sorted(dist_grid_codes)

def get_pyg_graphs(ouput_base_dir, grid_type, split='train'):
    dataset_path = os.path.join(ouput_base_dir, grid_type, split, f'dataset.pt')
    pyg_dataset = torch.load(dataset_path, weights_only=False)
    return pyg_dataset

def get_networkx_graph(data, include_features=True):
    if include_features:
        return to_networkx(data, node_attrs=['x', 'y'], edge_attrs=['edge_attr'], to_undirected='upper')
    else:
        return to_networkx(data, to_undirected='upper')

def get_pp_sources(ouput_base_dir, grid_type, split='train'):
    dataset_source_path = os.path.join(ouput_base_dir, grid_type, split, f'dataset_src.csv')
    sources = pd.read_csv(dataset_source_path, index_col=0)['src'].to_list()
    return sources

def get_cycle_lengths(nx_graph):
    cycles = nx.cycle_basis(nx_graph)
    cycle_lengths = [0]*len(nx_graph.nodes)
    for cycle in cycles:
        curr_length = len(cycle)
        for node in cycle:
            prev_shortest_length = cycle_lengths[node]
            if prev_shortest_length == 0 or curr_length < prev_shortest_length:
                cycle_lengths[node] = curr_length
    return cycle_lengths

def get_path_lengths_to_slack(nx_graph, slack_bus):
    paths = [len(path) - 1 for _, path in 
             sorted(nx.shortest_path(nx_graph, target=slack_bus).items())]
    return paths

def get_curvature_filtrations(nx_graph):
    pass

def add_augmented_features(dataset, cycles=False, path_lengths=False, curvature_filtrations=False):
    for data in dataset:
        nx_graph = get_networkx_graph(data)
        augmented_features = []

        if cycles:
            augmented_features.append(get_cycle_lengths(nx_graph))

        if path_lengths:
            slack_bus = -1
            for i, node in enumerate(data.x):
                if node[0] == 1:
                    slack_bus = i
                    break
            assert slack_bus != -1
            augmented_features.append(
                get_path_lengths_to_slack(nx_graph, slack_bus))
        
        if curvature_filtrations:
            augmented_features.append(get_curvature_filtrations(nx_graph))

        if augmented_features:
            augmented_features = np.vstack([augmented_features]).T
            data.x = torch.tensor(np.hstack([data.x, augmented_features]),
                                  dtype=torch.float32)
    return dataset