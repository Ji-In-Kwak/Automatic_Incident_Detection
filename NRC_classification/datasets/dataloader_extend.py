from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, remove_self_loops, add_self_loops, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
import torch
import torch.utils.data
import os.path as osp
import numpy as np
import pandas as pd
import torch
import networkx as nx
import random
from datasets.prepocessing_pyg import one_class_processing
from datasets.utils import load_new_data, MyDataset
from tqdm import tqdm
from datetime import datetime, timedelta, date
from sklearn.preprocessing import RobustScaler
import pickle



def traffic_loader_extend(args):
    
    train_loader_all, val_loader_all, test_loader_all = [], [], []

    for target_sid in args.dataset:
        if args.kfold == None:
            train = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/train.npz'.format(target_sid))
            val = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/val.npz'.format(target_sid))  
            test = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/test.npz'.format(target_sid))
        else:
            train = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/train{}.npz'.format(target_sid, args.kfold))
            val = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/val{}.npz'.format(target_sid, args.kfold))  
            test = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/test{}.npz'.format(target_sid, args.kfold))

        H = nx.read_gpickle("/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/sensor_graph.gpickle".format(target_sid))

        def pyg_dataset(H, dataset, labels, mode='train'):
            g_all = []
            print(mode, 'dataset')
            for i in tqdm(range(len(dataset))):  
                node_features = torch.FloatTensor(dataset[i, :, :].transpose(1,0)) ## (time, n_node)
                # node_features = node_features[list(H.nodes)]
                pyg_graph = from_networkx(H)
                # add self loop
                if args.self_loop == True:
                    pyg_graph.edge_index = add_self_loops(pyg_graph.edge_index)[0]
                if args.bidirect == True:
                    pyg_graph.edge_index = to_undirected(pyg_graph.edge_index) 
                
                pyg_graph.x = node_features
                pyg_graph.y = torch.tensor(labels[i])
                g_all.append(pyg_graph)
                
                # pyg_graph = Data(x=node_features, edge_index=edges, y=torch.tensor(labels[i]))
                # g_all.append(pyg_graph)
            return g_all
            
        train_dataset = pyg_dataset(H, train['x'], train['y'], 'train')
        val_dataset = pyg_dataset(H, val['x'], val['y'], 'val')
        test_dataset = pyg_dataset(H, test['x'], test['y'], 'test')

        bs = args.batch_size
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        train_loader_all.append(train_loader)
        val_loader_all.append(val_loader)
        test_loader_all.append(test_loader)
        
    return train_loader_all, val_loader_all, test_loader_all
