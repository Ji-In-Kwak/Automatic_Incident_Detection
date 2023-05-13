import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F

    
class GAT(torch.nn.Module):
    def __init__(self, n_layers, in_dim, n_hidden, out_dim, activation, drop_ratio):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.gat_layers.append(GATConv(in_dim, n_hidden))
        # hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(GATConv(n_hidden, out_dim))
        self.dropout = nn.Dropout(p=drop_ratio)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != 0:
                x = self.dropout(x)

        return F.log_softmax(x, dim=1)
    

class GAT_gc(torch.nn.Module):
    def __init__(self, n_layers, in_dim, n_hidden, out_dim, activation, drop_ratio, readout_type='sum', reverse=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_dim, n_hidden))
        # hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GATConv(n_hidden, n_hidden))
        # output layer
        self.outlayer = GATConv(n_hidden, out_dim)

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
        
        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)

        # return F.log_softmax(x, dim=1)
        return x