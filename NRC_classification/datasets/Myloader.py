from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse, remove_self_loops, add_self_loops
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



def loader(args):

    print("Loading your own dataset {}!!".format(args.dataset))
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)

    dataset_list, _ = load_new_data(args.dataset)
    print(len(dataset_list))            
    dataset = MyDataset(path, dataset_list)

    input_dim = dataset[0]['x'].shape[1]

    # select only normal dataset
    normal_dataset = [d for d in dataset if d['y'] == 0]
    abnormal_dataset = [d for d in dataset if d['y'] == 1]

    train_ix = int(len(normal_dataset)*0.7)
    train_dataset = normal_dataset[:train_ix]
    mix_dataset = normal_dataset[train_ix:] + abnormal_dataset
    random.shuffle(mix_dataset)
    val_dataset = mix_dataset[:int(len(mix_dataset)*0.5)]
    test_dataset = mix_dataset[int(len(mix_dataset)*0.5):]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim#, label_dim



def traffic_loader(args, target_sid, method='unsup'):

    dataset = args.dataset
        
    train_df = pd.read_csv('../data/{}/train_x.csv'.format(dataset), index_col=0)
    val_df = pd.read_csv('../data/{}/val_x.csv'.format(dataset), index_col=0)
    test_df = pd.read_csv('../data/{}/test_x.csv'.format(dataset), index_col=0)
    train_df.columns = train_df.columns.astype(int)
    val_df.columns = val_df.columns.astype(int)
    test_df.columns = test_df.columns.astype(int)
    
    
    train_label = pd.read_csv('../data/{}/train_y.csv'.format(dataset), index_col=0)
    val_label = pd.read_csv('../data/{}/val_y.csv'.format(dataset), index_col=0)
    test_label = pd.read_csv('../data/{}/test_y.csv'.format(dataset), index_col=0)

    H = nx.read_gpickle("../data/{}/sensor_graph.gpickle".format(dataset))

    def pyg_dataset(H, data_df, label_df, mode='train'):
        g_all = []
        print(mode, 'dataset')
        if mode == 'train':
            for i in tqdm(range(len(data_df)-48)):  
                node_features = data_df.iloc[i:i+24]
                node_features = node_features[list(H.nodes)]
                pyg_graph = from_networkx(H)
                # add self loop
                if args.self_loop == True:
                    pyg_graph.edge_index = add_self_loops(pyg_graph.edge_index)[0]
                # remove the incident case in training set
                if 1 in label_df.iloc[i:i+24]['label'].values:
                    continue
                ## make a small dataset by deleting recurrent congestion
                # elif (node_features.min(axis=0) < -1.5).any():
                #     continue
                else:
                    pyg_graph.x = torch.FloatTensor(node_features.T.values)
                    pyg_graph.y = torch.tensor([0])
                    g_all.append(pyg_graph)
        else:
            for i in tqdm(range(len(data_df)-24)): 
                node_features = data_df.iloc[i:i+24]
                node_features = node_features[list(H.nodes)]
                pyg_graph = from_networkx(H)
                if args.self_loop == True:
                    pyg_graph.edge_index = add_self_loops(pyg_graph.edge_index)[0]
                pyg_graph.x = torch.FloatTensor(node_features.T.values)
                pyg_graph.y = torch.tensor([label_df.iloc[i+24]])
                g_all.append(pyg_graph)

        return g_all
        
    if method == 'unsup':
        print('Unsupervised Learning')
        train_dataset = pyg_dataset(H, train_df, train_label, 'train')
    if method == 'semisup':
        print('Semi-supervised Learning')
        train_dataset = pyg_dataset(H, train_df, train_label, 'semi_train')
    val_dataset = pyg_dataset(H, val_df, val_label, 'val')
    test_dataset = pyg_dataset(H, test_df, test_label, 'test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def traffic_cls_loader(args, target_sid):
    print('cls loader!!!')

    dataset = args.dataset
        
    train_df = pd.read_csv('../data/{}/train_x.csv'.format(dataset), index_col=0)
    val_df = pd.read_csv('../data/{}/val_x.csv'.format(dataset), index_col=0)
    test_df = pd.read_csv('../data/{}/test_x.csv'.format(dataset), index_col=0)
    train_df.columns = train_df.columns.astype(int)
    val_df.columns = val_df.columns.astype(int)
    test_df.columns = test_df.columns.astype(int)
    
    
    train_label = pd.read_csv('../data/{}/train_y.csv'.format(dataset), index_col=0)
    val_label = pd.read_csv('../data/{}/val_y.csv'.format(dataset), index_col=0)
    test_label = pd.read_csv('../data/{}/test_y.csv'.format(dataset), index_col=0)

    H = nx.read_gpickle("../data/{}/sensor_graph.gpickle".format(dataset))

    def pyg_dataset(H, data_df, label_df, mode='train'):
        g_all = []
        print(mode, 'dataset')

        for i in tqdm(range(len(data_df)-48)):  
            node_features = data_df.iloc[i:i+24]
            node_features = node_features[list(H.nodes)]
            pyg_graph = from_networkx(H)
            # add self loop
            if args.self_loop == True:
                pyg_graph.edge_index = add_self_loops(pyg_graph.edge_index)[0]
            # remove the incident case in training set
            if label_df.iloc[i+24]['label'] == 0:    # classification for label 1 & 2 (recur & nonrecur)
                continue
            else:
                pyg_graph.x = torch.FloatTensor(node_features.T.values)
                pyg_graph.y = torch.tensor([label_df.iloc[i+24]['label']-1])
                g_all.append(pyg_graph)

        return g_all
        
    # if method == 'unsup':
    #     print('Unsupervised Learning')
    #     train_dataset = pyg_dataset(H, train_df, train_label, 'train')
    # if method == 'semisup':
    print('Semi-supervised Learning')
    train_dataset = pyg_dataset(H, train_df, train_label, 'train')
    val_dataset = pyg_dataset(H, val_df, val_label, 'val')
    test_dataset = pyg_dataset(H, test_df, test_label, 'test')

    # y_all = []
    # for g in train_dataset:
    #     y_all.append(g.y)
    # print(np.unique(torch.concat(y_all)))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader



# class StandardScaler():
#     """
#     Standard the input
#     """

#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def transform(self, data):
#         return (data - self.mean) / self.std

#     def inverse_transform(self, data):
#         return (data * self.std) + self.mean
    
def traffic_mtsc_loader(args):

    train = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/train.npz'.format(args.dataset))
    val = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/val.npz'.format(args.dataset))
    test = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/test.npz'.format(args.dataset))

    # try:
    #     with open('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/adj_mx.pkl'.format(args.dataset), 'rb') as f:
    #         sensor_file = pickle.load(f)
    # except UnicodeDecodeError as e:
    #     with open('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/adj_mx.pkl'.format(args.dataset), 'rb') as f:
    #         sensor_file = pickle.load(f, encoding='latin1')
    # except Exception as e:
    #     print('Unable to load data ', 'sensor_graph/adj_mx.pkl', ':', e)

    # sensor_ids, sensor_id_to_ind, adj_mx = sensor_file

    H = nx.read_gpickle("/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/sensor_graph.gpickle".format(args.dataset))

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

    return train_loader, val_loader, test_loader
