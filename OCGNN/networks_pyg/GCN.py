import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import torch.nn.functional as F

    
class GCN(torch.nn.Module):
    def __init__(self, n_layers, in_dim, n_hidden, out_dim, drop_ratio):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_dim, n_hidden))
        # hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(GCNConv(n_hidden, out_dim))
        self.dropout = nn.Dropout(p=drop_ratio)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != 0:
                x = self.dropout(x)

        return F.log_softmax(x, dim=1)
    

class GCN_gc(torch.nn.Module):
    def __init__(self, n_layers, in_dim, n_hidden, out_dim, activation, drop_ratio, readout_type='sum', reverse=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(in_dim, n_hidden))
        # hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(n_hidden, n_hidden))
        # output layer
        self.outlayer = GCNConv(n_hidden, out_dim)

        
        self.dropout = nn.Dropout(p=drop_ratio)

        # readout layer
        self.readout = readout_type

        self.reverse = reverse

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.reverse:
            edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
            edge_index = edge_index_rev

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != 0:
                x = self.dropout(x)
            x = F.relu(x)
        x = self.outlayer(x, edge_index)
        
        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)

        # return F.log_softmax(x, dim=1)
        return x
    

class GCN_traffic(torch.nn.Module):
    def __init__(self, n_layers, in_dim, n_hidden, out_dim, activation, drop_ratio, readout_type):
        super().__init__()
#         self.layers = nn.ModuleList()
#         # input layer
#         self.layers.append(GCNConv(in_dim, n_hidden))
#         # hidden layer
#         for i in range(n_layers - 1):
#             self.layers.append(GCNConv(n_hidden, n_hidden))
#         # output layer
#         self.outlayer = GCNConv(n_hidden, out_dim)


        self.reverse_layers = nn.ModuleList()
        # input layer
        self.reverse_layers.append(GCNConv(in_dim, n_hidden))
        # hidden layer
        for i in range(n_layers - 1):
            self.reverse_layers.append(GCNConv(n_hidden, n_hidden))
        # output layer
        self.reverse_outlayer = GCNConv(n_hidden, out_dim)

        self.mlp_layer = nn.Linear(out_dim*2, out_dim)

        self.dropout = nn.Dropout(p=drop_ratio)

        # readout layer
        # self.readout = readout_type
        self.readout = 'mean'

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_rev, edge_index_rev = data.x, torch.stack([edge_index[1], edge_index[0]])

#         # original graph
#         for i, layer in enumerate(self.layers):
#             x = layer(x, edge_index)
#             if i != 0:
#                 x = self.dropout(x)
#             x = F.relu(x)
#         x = self.outlayer(x, edge_index)

        # reverse graph
        for i, layer in enumerate(self.reverse_layers):
            x_rev = layer(x_rev, edge_index_rev)
            if i != 0:
                x_rev = self.dropout(x_rev)
            x_rev = F.relu(x_rev)
        x_rev = self.reverse_outlayer(x_rev, edge_index_rev)

#         h = torch.cat([x, x_rev], dim=1)
#         h = self.mlp_layer(h)
        h = x_rev

#         if self.readout == 'mean':
#         h = global_mean_pool(h, batch)
        h = global_add_pool(h, batch)

        # return F.log_softmax(x, dim=1)
        return h