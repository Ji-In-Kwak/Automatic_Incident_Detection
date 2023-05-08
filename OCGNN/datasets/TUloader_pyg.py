from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch
import torch.utils.data
import numpy as np
import torch
import dgl
import networkx as nx


# add self loop for TU dataset, other datasets haven't been tested. 
def pre_process(args,dataset):

    for i in range(len(dataset)):
        #print(dataset.graph_lists[i])
        #make labels become 0 or 1, other label is not our need.
        #dataset.graph_lists[i].ndata
        # normal_idx=torch.where(dataset.graph_lists[i].ndata['node_labels']==args.normal_class)[0]
        # abnormal_idx=torch.where(dataset.graph_lists[i].ndata['node_labels']!=args.normal_class)[0]
        normal_idx=torch.where(dataset[i]['y']==args.normal_class)[0]
        abnormal_idx=torch.where(dataset[i]['y']!=args.normal_class)[0]
        dataset[i]['y'][normal_idx]=0
        dataset[i]['y'][abnormal_idx]=1
        

        if args.self_loop:
            add_self_loops(dataset[i].edge_index)
            # g=dgl.transform.add_self_loop(dataset.graph_lists[i])
            # g.ndata.update(dataset.graph_lists[i].ndata)
            # dataset.graph_lists[i]=g

        #print(dataset.graph_lists[i].ndata['node_labels'].max())
        #print(dataset.graph_lists[i])
    return dataset

def loader(args):

    # dataset = tu.TUDataset(name=args.dataset)
    dataset = TUDataset(root=f'/tmp/{args.dataset}', name=args.dataset)
    print(dataset)
    print(dataset[0])
    
    train_size = int(0.6 * len(dataset))
    #train_size=16
    test_size = int(0.25 * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)
    
    dataset = pre_process(args,dataset)

    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, (train_size, val_size, test_size))
    train_loader = prepare_dataloader(dataset_train, args, train=True)
    val_loader = prepare_dataloader(dataset_val, args, train=False)
    test_loader = prepare_dataloader(dataset_test, args, train=False)

    # input_dim,label_dim, max_num_node = dataset.statistics() #I rewrited the code of dgl.tu
    input_dim = dataset[0]['x'].shape[1]
    
    print("++++++++++STATISTICS ABOUT THE DATASET")
    print("dataset feature dimension is", input_dim)
    # print("dataset label dimension is", label_dim)
    # print("the max num node is", max_num_node)
    print("number of graphs is", len(dataset))


    return train_loader, val_loader, test_loader, input_dim,# label_dim


def prepare_dataloader(dataset, args, train=False, pre_process=None):
    '''
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    '''
    if train:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    if pre_process:
        pre_process(dataset, args)

    # dataset.set_fold(fold)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=args.batch_size,
                                       shuffle=shuffle,
                                       collate_fn=batching_graph,
                                       drop_last=drop_last,
                                       num_workers=args.n_worker)

def batching_graph(batch):
    '''
    for dataset batching
    transform ndata to tensor (in gpu is available)
    '''
    # graphs, labels = map(list, zip(*batch))
    graphs, l
    #cuda = torch.cuda.is_available()

    # batch graphs and cast to PyTorch tensor
    for graph in graphs:
        for (key, value) in graph.ndata.items():
            graph.ndata[key] = value.float()
    batched_graphs = dgl.batch(graphs)

    # cast to PyTorch tensor
    batched_labels = torch.LongTensor(np.array(labels))

    return batched_graphs, batched_labels