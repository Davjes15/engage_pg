import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ARMAConv

# =============================================================================
#                       Experiment models
# =============================================================================

# ARMA GNN class - Adaptation from paper.
# J. B. Hansen, S. N. Anfinsen, and F. M. Bianchi,
# “Power Flow Balancing With Decentralized Graph Neural Networks,”
# IEEE Transactions on Power Systems, vol. 38, no. 3, pp. 2423–2433, May 2023,
# doi: 10.1109/TPWRS.2022.3195301.
class ARMA_GNN(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.input_dim = input_dim # from kwargs
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.leakyReLU_small = nn.LeakyReLU(negative_slope=0.0)

        # Pre-processing layers
        self.predense1_node = nn.Linear(self.input_dim, 64)
        self.predense2_node = nn.Linear(64, 64)
        self.predense1_edge = nn.Linear(4, 16)
        self.predense2_edge = nn.Linear(16, 1)

        # ARMA layers
        self.arma = ARMAConv(64, 64, num_stacks=8, num_layers=4, shared_weights=False, act=self.leakyReLU, dropout=0.0, bias=True)

        # Post-processing layer
        self.postdense1 = nn.Linear(64, 64)
        self.postdense2 = nn.Linear(64, 64)

        # Output layer
        self.readout = nn.Linear(64, 4)

    def forward(self, data):
        x = torch.nan_to_num(data.x, nan=0.0) # dim=(N, self.input_dim)
        edge_index = data.edge_index # dim=(2, 2E)
        # edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0) # dim=(2E, 4)
        
        node_emb = self.leakyReLU(self.predense1_node(x))
        node_emb = self.leakyReLU(self.predense2_node(node_emb))

        # edge_emb = self.leakyReLU(self.predense1_edge(edge_attr))
        # edge_emb = self.leakyReLU(self.predense2_edge(edge_emb))
        # edge_emb = edge_emb.reshape((-1,))

        # node_emb = self.arma(node_emb, edge_index, edge_weight=edge_emb)

        # Edge_emb causes nan so do without for now.
        node_emb = self.arma(node_emb, edge_index, edge_weight=None)

        node_emb = self.leakyReLU(self.postdense1(node_emb))
        node_emb = self.leakyReLU(self.postdense2(node_emb))

        return self.readout(node_emb)


# GCN class - custom GCN class w/ similar number of params as ARMA_GNN.
class GCN(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        # Constants
        self.input_dim = input_dim
        self.num_gcn_layers = 8
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.leakyReLU_small = nn.LeakyReLU(negative_slope=0.005)

        # Pre-processing layers
        self.predense1_node = nn.Linear(self.input_dim, 64)
        self.predense2_node = nn.Linear(64, 64)
        self.predense1_edge = nn.Linear(4, 16)
        self.predense2_edge = nn.Linear(16, 1)

        self.gcn_layers = nn.ModuleList([
            GCNConv(64, 64, normalize=True) for i in range(self.num_gcn_layers)
        ])

        # Post-processing layer
        self.postdense1 = nn.Linear(64, 64)
        self.postdense2 = nn.Linear(64, 64)

        # Output layer
        self.readout = nn.Linear(64, 4)

    def forward(self, data):
        x = torch.nan_to_num(data.x, nan=0.0) # dim=(N, self.input_dim)
        edge_index = data.edge_index # dim=(2, 2E)
        edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0) # dim=(2E, 4)
        
        node_emb = self.leakyReLU(self.predense1_node(x))
        node_emb = self.leakyReLU(self.predense2_node(node_emb))

        edge_emb = self.leakyReLU(self.predense1_edge(edge_attr))
        # Using a leaky relu with too large of negative_slope can lead to
        # sqrt of a negative number in GCNConv, so we use different leakyReLU
        # in last step.
        edge_emb = self.leakyReLU_small(self.predense2_edge(edge_emb))
        edge_emb = edge_emb.reshape((-1,))

        for i, layer in enumerate(self.gcn_layers):
            node_emb = self.leakyReLU(layer(x=node_emb,
                                            edge_index=edge_index,
                                            edge_weight=edge_emb))

        node_emb = self.leakyReLU(self.postdense1(node_emb))
        node_emb = self.leakyReLU(self.postdense2(node_emb))

        return self.readout(node_emb)
