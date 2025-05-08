# Generating DG Data using Simbench and Powerdata-gen
# 
# [1] https://github.com/e2nIEE/simbench  
# [2] https://github.com/bdonon/powerdata-gen

### Load dependencies, including the data generator library

import sys, os
DATA_GEN_PATH = os.path.abspath('powerdata-gen/')
sys.path.append(DATA_GEN_PATH)
import powerdata_gen
# Reset path
sys.path.pop()

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Data

import time
import logging
import argparse
from copy import deepcopy

from graph_utils import get_dist_grid_codes

### Helper functions for script arguments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry_run",
        action="store_true"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


### Helper functions to load pandapower grids from simbench

def create_output_dir():
    identifier = time.strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = os.path.join('outputs', identifier)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_pandapower_grid_to_json(sb_code: str, filename: str):
    net = sb.get_simbench_net(sb_code)
    pp.to_json(net, filename)
    return filename


### Helper functions for extracting node features and edge features

def get_node_features(net):
    # List of bus features
    #   x: np.array([Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree])
    #   y: np.array([p_mw, q_mvar, vm_pu, va_degree])
    #
    node_features_x, node_features_y = [], [] # map from bus_id to features
    for bus_id in net.bus.index:
        # (Slack?, PV?, PQ?)
        bus_type = (0, 0, 1)

        gens = net.gen.loc[net.gen['bus'] == bus_id]
        if len(gens) > 0:
            bus_type = (0, 1, 0)

        slack = net.ext_grid.loc[net.ext_grid['bus'] == bus_id,
                        ['vm_pu', 'va_degree']]
        if len(slack) > 0:
            assert len(gens) == 0, ('PV and Swing generators cannot be placed'
                                    ' on the same bus. This is because they'
                                    ' will both try to control the bus voltage.')
            bus_type = (1, 0, 0)
        
        # net.res_bus should already take into account all the components that
        # contribute to these four bus parameters so we do not have to do this
        # again (ex. loads, sgens, gens, storages, ext_grid, etc.).
        features = net.res_bus.loc[bus_id, ['p_mw', 'q_mvar', 'vm_pu', 'va_degree']]
        masked_features = features.copy()
        if bus_type[0]:
            masked_features['p_mw'] = np.nan
            masked_features['q_mvar'] = np.nan
        elif bus_type[1]:
            masked_features['q_mvar'] = np.nan
            masked_features['va_degree'] = np.nan
        else:
            masked_features['vm_pu'] = np.nan
            masked_features['va_degree'] = np.nan

        node_features_x.append(np.append(bus_type, masked_features.values))
        node_features_y.append(features.values)
    
    return np.array(node_features_x), np.array(node_features_y)

def get_edge_features(net):
    # List of edge features
    #   e: np.array([trafo?, r_pu, x_pu, sc_voltage])

    def get_line_features(net):
        # Undirected graph so need to add both directions to edge_index.
        edge_index = net.line.loc[:, ['from_bus', 'to_bus',
                                      'to_bus', 'from_bus']].values
        # Use .reshape to change shape from (E, 4) to (2E, 2), where E is num edges.
        # Transpose to make into proper (2, 2E format).
        edge_index = edge_index.reshape(-1, 2).T

        r = net.line['r_ohm_per_km'].values * net.line['length_km'].values
        x = net.line['x_ohm_per_km'].values * net.line['length_km'].values

        # We convert the r,x values into per unit (p.u.) to simplify calculations
        # and ensure consistency across the network. To do this, we divide r, x by
        # the base impedance. Therefore z = vn_kv**2/sn_mva, where vn_kv is rated
        # voltage and sn_mva is reference apparent power.
        # Note: vn_kv be the same for every bus except ext_grid, but this is safer.
        vn_kv = net.bus.loc[net.line['to_bus'], ['vn_kv']].values.reshape(-1)
        z = np.square(vn_kv) / net.sn_mva
        r_pu = r / z
        x_pu = x / z

        # Similarly, due to undirected graph, the edge features need to be repeated
        # twice, once for each respective connection present in the COO matrix.
        r_pu = r_pu.repeat(2)
        x_pu = x_pu.repeat(2)

        # Add encoding for a line and pad with nan to account for missing short
        # circuit voltage.
        e = edge_index.shape[1] # b/c coo matrix
        edge_features = np.vstack([np.zeros(e),         # trafo?
                                   r_pu,                # r_pu
                                   x_pu,                # x_pu
                                   np.nan*np.ones(e)    # sc_voltage
                                   ]).T

        return edge_index, edge_features

    def get_trafo_features(net):
        # Similar to get_line_features.
        edge_index = net.trafo.loc[:, ['hv_bus', 'lv_bus',
                                       'lv_bus', 'hv_bus']].values
        edge_index = edge_index.reshape(-1, 2).T

        # Impedance calculated as shown in pandapower docs:
        # https://pandapower.readthedocs.io/en/v2.14.11/elements/trafo.html#impedance-values
        # where vk_percent is short-circuit voltage and vkr_percent is the real
        # part of short-circuit voltage (%).
        z_pu = (net.trafo['vk_percent'].values / 100)*(net.sn_mva / net.trafo['sn_mva'].values)
        r_pu = (net.trafo['vkr_percent'].values / 100)*(net.sn_mva / net.trafo['sn_mva'].values)
        x_pu = np.sqrt(np.square(z_pu) - np.square(r_pu))

        # Add relative short-circuit voltage as additional feature.
        sc_voltage = net.trafo['vk_percent'].values
        
        # Repeat the features (to match edge_index) and create feature matrix.
        e = edge_index.shape[1] # b/c coo matrix
        edge_features = np.vstack([np.ones(e),              # trafo?
                                   r_pu.repeat(2),          # r_pu
                                   x_pu.repeat(2),          # x_pu
                                   sc_voltage.repeat(2)     # sc_voltage
                                   ]).T

        return edge_index, edge_features
    
    A_line, E_line = get_line_features(net)
    A_trafo, E_trafo = get_trafo_features(net)
    
    # Combine and return the line and trafo features.
    A = np.hstack([A_line, A_trafo])
    E = np.vstack([E_line, E_trafo])

    # Sometimes bus ids are higher than the number of nodes. This can mess up
    # the adjacency matrix (edge_index) so we need to remap back to smaller ids.
    # We assume the graph is fully connected, so every node id exists at least
    # once in the edge_index.
    unique_nodes = set(A[0])
    remapping = dict(zip(sorted(unique_nodes), range(len(unique_nodes))))
    applyall = np.vectorize(lambda x: remapping[x])
    A = applyall(A)
    
    return A, E


### Generate Grids

if __name__ == '__main__':
    args = parse_args()

    TEST_GENERATION = args.dry_run
    size = args.size

    # Simbench codes for distribution grids
    sb_codes = get_dist_grid_codes(args.scenario)
    sb_codes = sb_codes[0:1] if TEST_GENERATION else sb_codes
    dataset_split = [2, 0, 0] if TEST_GENERATION else [size, 0, 0]

    # Setup input directory
    input_dir = 'inputs/'
    os.makedirs(input_dir, exist_ok=True)

    # Convert simbench codes to json files, if not already done.
    filenames = []
    for code in sb_codes:
        f = os.path.join(input_dir, f'{code}.json')
        if not os.path.exists(f):
            save_pandapower_grid_to_json(code, f)
        filenames.append(f)

    # Load a base config file and change adjust parameters.
    cfg = OmegaConf.load('base_gen_config.yaml')
    cfg.n_train, cfg.n_val, cfg.n_test = dataset_split
    cfg.seed = 12

    # Set up logger (for powerdata-gen)
    log = logging.getLogger(__name__)

    # Create output directory for all data from this run, loop through ref grids, 
    # generate new grids for each ref, save them to subdir of output dir.
    output_dir = create_output_dir()
    print(f'Output directory: {output_dir}\n')
    generated_grid_base_dirs = []
    for code, f in list(zip(sb_codes, filenames)):
        save_path = os.path.join(output_dir, code)
        os.makedirs(save_path, exist_ok=True)
        cfg.default_net_path = f
        powerdata_gen.build_datasets(cfg.default_net_path,
                                     save_path,
                                     log,
                                     cfg.n_train,
                                     cfg.n_val,
                                     cfg.n_test,
                                     cfg.keep_reject,
                                     cfg.sampling,
                                     cfg.powerflow,
                                     cfg.filtering,
                                     cfg.seed)
        generated_grid_base_dirs.append(save_path)


    # Create PyTorch datasets using the generated grids

    for dir in generated_grid_base_dirs:
        generated_grid_dir = os.path.join(dir, 'train')
        generated_grids = os.listdir(generated_grid_dir)
        # list[outputs/<identifier>/<sb_code>/<train|test|val>/sample_<N>.json]
        generated_grid_files = [os.path.join(generated_grid_dir, f) for f in generated_grids]

        dataset_filename = os.path.join(generated_grid_dir,
                                        f'dataset.pt')
        
        dataset_source = os.path.join(generated_grid_dir,
                                        f'dataset_src.csv')

        # If we have already created this dataset, skip.
        if os.path.exists(dataset_filename) and os.path.exists(dataset_source):
            continue
        dataset = []
        srcs = []

        for f in generated_grid_files:
            if f.split('.')[-1] != 'json':
                # There could be non json files that exist, so skip them.
                continue
            net = pp.from_json(f)

            X_i, Y_i = get_node_features(net)
            A_i, E_i = get_edge_features(net)

            # Run dc_pf, such that we can use this later and do not have to compute every time.

            # Load the source network
            net = deepcopy(net)
            # Run dc pf
            pp.rundcpp(net)
            # Put this in correct format to match the true data and get np array.
            np_dc_pf = net.res_bus[['p_mw', 'q_mvar', 'vm_pu', 'va_degree']].values
            # Convert to tensor and replace nan (q_mwar) with 0.
            dc_pf = torch.nan_to_num(torch.Tensor(np_dc_pf), nan=0.0)

            # Data dimensions
            #   x: (N, 7), where 7 are [Slack?, PV?, PQ?, p_mw, q_mvar, vm_pu, va_degree]
            #   edge_index: (2, 2E)
            #   edge_attr: (2E, 5), where 5 are [trafo?, r_pu, x_pu, length, sc_voltage]
            #   y: (N, 4), where 4 are [p_mw, q_mvar, vm_pu, va_degree]
            #   dc_pf: (N, 4), where 4 are [p_mw, q_mvar, vm_pu, va_degree]
            #

            dataset.append(
                Data(x=torch.tensor(X_i, dtype=torch.float32),
                    edge_index=torch.tensor(A_i, dtype=torch.int64),
                    edge_attr=torch.tensor(E_i, dtype=torch.float32),
                    y=torch.tensor(Y_i, dtype=torch.float32),
                    dc_pf=dc_pf)
            )
            srcs.append(f)
        
        print('Saving dataset in', dataset_filename, end='... ')
        torch.save(dataset, dataset_filename)
        print('completed')
        print('Saving source list in', dataset_source,  end='... ')
        pd.DataFrame(srcs, columns=['src']).to_csv(dataset_source)
        print('completed')




