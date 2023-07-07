import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import torch.nn.functional as F
import torch.nn.init as init
import math

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x
    
class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result
    
class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * Residual Connection *
    #        |                                |
    #        |    |--->--- CasualConv2d ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- CasualConv2d --- Sigmoid ---|                               
    #
    
    #param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        # self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                # GLU was first purposed in
                # *Language Modeling with Gated Convolutional Networks*.
                # URL: https://arxiv.org/abs/1612.08083
                # Input tensor X is split by a certain dimension into tensor X_a and X_b.
                # In PyTorch, GLU is defined as X_a ⊙ Sigmoid(X_b).
                # URL: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # (x_p + x_in) ⊙ Sigmoid(x_q)
                x = torch.mul((x_p + x_in), self.sigmoid(x_q))

            else:
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                x = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        
        elif self.act_func == 'leaky_relu':
            x = self.leaky_relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        
        return x
    

# class GraphConv(nn.Module):
#     def __init__(self, c_in, c_out, gso, bias):
#         super(GraphConv, self).__init__()
#         self.c_in = c_in
#         self.c_out = c_out
#         self.gso = gso
#         self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(c_out))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             init.uniform_(self.bias, -bound, bound)

#     def forward(self, x):
#         #bs, c_in, ts, n_vertex = x.shape
#         x = torch.permute(x, (0, 2, 3, 1))

#         first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
#         second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

#         if self.bias is not None:
#             graph_conv = torch.add(second_mul, self.bias)
#         else:
#             graph_conv = second_mul
        
#         return graph_conv

# class GraphConvLayer(nn.Module):
#     def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
#         super(GraphConvLayer, self).__init__()
#         self.graph_conv_type = graph_conv_type
#         self.c_in = c_in
#         self.c_out = c_out
#         self.align = Align(c_in, c_out)
#         self.Ks = Ks
#         self.gso = gso
#         if self.graph_conv_type == 'cheb_graph_conv':
#             self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
#         elif self.graph_conv_type == 'graph_conv':
#             self.graph_conv = GraphConv(c_out, c_out, gso, bias)

#     def forward(self, x):
#         x_gc_in = self.align(x)
#         if self.graph_conv_type == 'cheb_graph_conv':
#             x_gc = self.cheb_graph_conv(x_gc_in)
#         elif self.graph_conv_type == 'graph_conv':
#             x_gc = self.graph_conv(x_gc_in)
#         x_gc = x_gc.permute(0, 3, 1, 2)
#         x_gc_out = torch.add(x_gc, x_gc_in)

#         return x_gc_out

# class STConvBlock(nn.Module):
#     # STConv Block contains 'TGTND' structure
#     # T: Gated Temporal Convolution Layer (GLU or GTU)
#     # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
#     # T: Gated Temporal Convolution Layer (GLU or GTU)
#     # N: Layer Normolization
#     # D: Dropout

#     def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
#         super(STConvBlock, self).__init__()
#         self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
#         self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
#         self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
#         self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=droprate)

#     def forward(self, x):
#         x = self.tmp_conv1(x)
#         x = self.graph_conv(x)
#         x = self.relu(x)
#         x = self.tmp_conv2(x)
#         x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         x = self.dropout(x)

#         return x


class STblock(torch.nn.Module):
    def __init__(self, last_block_channel, channels, heads, ts_dim, activation, negative_slope, drop_ratio, bias=True):
        super().__init__()
        # self.stblocks = nn.ModuleList()
        # self.layers = nn.ModuleList()

        # channels = [4, 16, out_dim]
        self.channels = channels
        self.ts_dim = ts_dim
        self.kernel_size = 3  ### ????
        # [bs, c_in, ts, n_vertex]
        self.tmp_conv1 = TemporalConvLayer(self.kernel_size, last_block_channel, channels[0], 'glu')
        # self.graph_conv = GCNConv(channels[0], channels[1], bias=bias)
        self.graph_conv = GATConv(channels[0], channels[1], heads, negative_slope=negative_slope, dropout=drop_ratio, bias=bias)
        self.tmp_conv2 = TemporalConvLayer(self.kernel_size, channels[1]*heads, channels[2], 'glu')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_ratio)



    def forward(self, x, edge_index, batch):

        # print("input = ", x.shape)   # [c0, n_nodes, ts]
        # make [1, 1, ts, n_nodes]
        x = x.unsqueeze(0)
        x = x.permute(0,1,3,2)
        x = self.tmp_conv1(x)
        # print("temp1 = ", x.shape)   # [1, c1, ts-k+1, n_nodes]

        gcn_input = x.permute(0, 2, 3, 1)
        gcn_out = []
        for i in range(self.ts_dim-self.kernel_size+1):
            tmp = self.graph_conv(gcn_input[0, i, :, :], edge_index)
            gcn_out.append(self.relu(tmp))
        x = torch.stack(gcn_out)
        # print("GCN = ", x.shape)  # [ts-k+1, n_nodes, c2]

        x = x.unsqueeze(0).permute(0,3,1,2)   # [1, c2, ts-k+1, n_nodes]
        x = self.tmp_conv2(x)
        # print("temp2 = ", x.shape)  # [1, c3, ts-2k+2, n_nodes]
        x = self.dropout(x)

        return x


class STGAT(torch.nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, heads, activation, drop_ratio, negative_slope, readout_type, reverse=False, bias=True):
        super().__init__()

        self.stblocks = nn.ModuleList()
        last_block_channel = 1
        channel = [1, 2, 4]
        kernel_size = 3
        ts_dim = in_dim

        for i in range(n_layers):
            # if i == n_layers-1:
            #     channel[-1] = out_dim
            ts_dim = in_dim - 2*kernel_size*i + 2*i
            self.stblocks.append(STblock(last_block_channel, channel, heads[i], ts_dim, 'relu', negative_slope, drop_ratio, bias))
            last_block_channel = channel[-1]
            channel = [2*c for c in channel]
        ts_dim = in_dim - 2*kernel_size*n_layers + 2*n_layers
        # self.outlayer = GCNConv(last_block_channel*ts_dim, out_dim, bias=bias)
        self.outlayer = GATConv(last_block_channel*ts_dim, out_dim, heads[-1], negative_slope=negative_slope, dropout=drop_ratio, bias=bias)

        # readout layer
        self.readout = readout_type

        # reverse adjacency matrix
        self.reverse = reverse
            
            

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.reverse:
            edge_index_rev = torch.stack([edge_index[1], edge_index[0]])
            edge_index = edge_index_rev

        x = x.unsqueeze(0)
        
        for i, block in enumerate(self.stblocks):
            x = block(x, edge_index, batch)
            x = x.squeeze(0).permute(0, 2, 1)

        x = x.permute(1,0,2).reshape(len(batch), -1) 
        # print('last embed = ', x.shape)
        x = self.outlayer(x, edge_index)


        if self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)

        return x



