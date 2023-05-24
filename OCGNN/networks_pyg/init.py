import torch.nn.functional as F
from networks_pyg.GCN import GCN, GCN_gc, GCN_traffic
from networks_pyg.GAT import GAT_gc
# from networks_pyg.GAE import GAE
# from networks_pyg.GIN import GIN
from networks_pyg.GraphSAGE import GraphSAGE, GraphSAGE_gc
from networks_pyg.STGCN import STGCN
from networks_pyg.GatedGCN import GatedGCN


def init_model(args,input_dim):
    # create GCN model
    if args.module== 'GCN':
        model = GCN(
                args.n_layers,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                F.relu,
                args.dropout)
    if args.module== 'GCN_gc':
        model = GCN_gc(
                args.n_layers,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                F.relu,
                args.dropout,
                readout_type=args.pooling,
                reverse=args.reverse) # mean, sum, max, #sagpool
    if args.module== 'GAT_gc':
        model = GAT_gc(
                args.n_layers,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                heads=([8] * args.n_layers) + [1],
                activation=F.relu,
                drop_ratio=args.dropout,
                negative_slope=0.2,
                readout_type=args.pooling,
                reverse=args.reverse)
    # if args.module== 'GCN_traffic':
    #     model = GCN_traffic(
    #             args.n_layers,
    #             input_dim,
    #             args.n_hidden*2,
    #             args.n_hidden,
    #             F.relu,
    #             args.dropout,
    #             readout_type='mean') # mean, sum, max, #sagpool
    if args.module== 'GraphSAGE':
        model = GraphSAGE(
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout,
                aggregator_type='SoftmaxAggregation') #mean,pool,lstm,gcn 使用pool做多图学习有大问题阿
    if args.module== 'GraphSAGE_gc':
        model = GraphSAGE_gc(
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout,
                aggregator_type='SoftmaxAggregation',
                readout_type=args.pooling,
                reverse=args.reverse) 

    if args.module == 'STGCN':
        model = STGCN(
                args.n_layers, 
                input_dim, 
                args.n_hidden*2, 
                args.n_hidden, 
                F.relu, 
                args.dropout, 
                readout_type=args.pooling,
                reverse=args.reverse)


#     if args.module== 'GIN':
#         model = GIN(num_layers=args.n_layers, 
#                     num_mlp_layers=2, #1 means linear model.
#                     input_dim=input_dim, 
#                     hidden_dim=args.n_hidden*2,
#                     output_dim=args.n_hidden, 
#                     final_dropout=args.dropout, 
#                     learn_eps=False, 
#                     graph_pooling_type="sum",
#                     neighbor_pooling_type="sum")
#     if args.module== 'GAE':
#         model = GAE(None,
#                 input_dim,
#                 n_hidden=args.n_hidden*2,
#                 n_classes=args.n_hidden,
#                 n_layers=args.n_layers,
#                 activation=F.relu,
#                 dropout=args.dropout)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
    if cuda:
        model.cuda()

    print(f'Parameter number of {args.module} Net is: {count_parameters(model)}')

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)