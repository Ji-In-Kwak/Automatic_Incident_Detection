import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, bias=True))
        self.layers.append(nn.ReLU())
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, bias=True))
            self.layers.append(nn.ReLU())
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, bias=True)) # activation None

        self.dropout = nn.Dropout(p=dropout)

        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
        x = self.out_layers(x, edge_index)
        
        return x
    

class GraphSAGE_gc(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 readout_type,
                 reverse=False):
        # super(GraphSAGE, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, bias=True))
        # self.layers.append(nn.ReLU())
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, bias=True))
            # self.layers.append(nn.ReLU())
        # output layer
        self.outlayer = SAGEConv(n_hidden, n_classes, aggregator_type, bias=True) # activation None

        self.dropout = nn.Dropout(p=dropout)

        # readout layer
        self.readout = readout_type

        # reverse adjacency matrix
        self.reverse = reverse

        
    def forward(self, data, rev=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.reverse:
            edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
            edge_index = edge_index_rev

        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.outlayer(x, edge_index)

        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
            
        return x
    
