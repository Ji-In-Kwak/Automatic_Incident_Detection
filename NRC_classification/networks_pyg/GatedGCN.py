import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import torch.nn.functional as F

    

class GatedGCN(torch.nn.Module):
    def __init__(self, n_layers, in_dim, n_hidden, out_dim, activation, drop_ratio, readout_type, reverse=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GatedGraphConv(n_hidden, in_dim))
        # # hidden layer
        # for i in range(n_layers - 1):
        #     self.layers.append(GatedGraphConv(n_hidden, n_hidden))
        # output layer
        self.outlayer = GatedGraphConv(n_hidden, out_dim)

        
        self.dropout = nn.Dropout(p=drop_ratio)

        # readout layer
        self.readout = readout_type

        self.reverse = reverse

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.reverse:
            edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
            edge_index = edge_index_rev

        # print(x.shape)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != 0:
                x = self.dropout(x)
            x = F.relu(x)
        # print(x.shape)
        x = self.outlayer(x, edge_index)
        # print(x.shape)
        x = global_add_pool(x, batch)

        # return F.log_softmax(x, dim=1)
        return x
    
