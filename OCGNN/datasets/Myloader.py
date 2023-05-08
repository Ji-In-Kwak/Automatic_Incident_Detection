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



def traffic_loader(args, target_sid):
        
    train_df = pd.read_csv('../data/{}/train_x.csv'.format(target_sid), index_col=0)
    val_df = pd.read_csv('../data/{}/val_x.csv'.format(target_sid), index_col=0)
    test_df = pd.read_csv('../data/{}/test_x.csv'.format(target_sid), index_col=0)
    train_df.columns = train_df.columns.astype(int)
    val_df.columns = val_df.columns.astype(int)
    test_df.columns = test_df.columns.astype(int)
    
    
    train_label = pd.read_csv('../data/{}/train_y.csv'.format(target_sid), index_col=0)
    val_label = pd.read_csv('../data/{}/val_y.csv'.format(target_sid), index_col=0)
    test_label = pd.read_csv('../data/{}/test_y.csv'.format(target_sid), index_col=0)

    H = nx.read_gpickle("../data/{}/sensor_graph.gpickle".format(target_sid))

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
        
    train_dataset = pyg_dataset(H, train_df, train_label, 'train')
    val_dataset = pyg_dataset(H, val_df, val_label, 'val')
    test_dataset = pyg_dataset(H, test_df, test_label, 'test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
def simulation_loader(args):

    data = {}
    for category in ['train', 'val', 'test']:
        # train = np.load('/media/usr/SSD/jiin/naver/data/METR-LA/train.npz')
        # val = np.load('/media/usr/SSD/jiin/naver/data/METR-LA/val.npz')
        # test = np.load('/media/usr/SSD/jiin/naver/data/METR-LA/test_anomaly.npz')
        if category != 'train':
            dataset = np.load('/media/usr/SSD/jiin/naver/data/METR-LA/{}.npz'.format(category+'_anomaly'))
        else:
            dataset = np.load('/media/usr/SSD/jiin/naver/data/METR-LA/{}.npz'.format(category))
        data[category+'_x'] = dataset['x']
        data[category+'_y'] = dataset['y']
    scaler = StandardScaler(mean = data['train_x'][:, :, :, 0].mean(), std=data['train_x'][:, :, :, 0].std())
    for category in ['train', 'val', 'test']:
        data[category+'_x'][...,0] = scaler.transform(data[category+'_x'][..., 0])


    # train_df = train_df.fillna(0)
    # val_df = val_df.fillna(0)
    # test_df = test_df.fillna(0)

    try:
        with open('/media/usr/SSD/jiin/naver/data/METR-LA/sensor_graph/adj_mx.pkl', 'rb') as f:
            sensor_file = pickle.load(f)
    except UnicodeDecodeError as e:
        with open('/media/usr/SSD/jiin/naver/data/METR-LA/sensor_graph/adj_mx.pkl', 'rb') as f:
            sensor_file = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', 'sensor_graph/adj_mx.pkl', ':', e)

    sensor_ids, sensor_id_to_ind, adj_mx = sensor_file

    def pyg_dataset(adj_mx, dataset, labels, mode='train'):
        g_all = []
        print(mode, 'dataset')
        if (mode == 'train'):
            for i in tqdm(range(len(dataset))):  
                node_features = torch.FloatTensor(dataset[i, :, :, 0]).transpose(1,0) ## (time, n_node)
                edges = dense_to_sparse(torch.Tensor(adj_mx))[0]
                pyg_graph = Data(x=node_features, edge_index=edges, y=torch.tensor([0]))
                g_all.append(pyg_graph)
        else:
            for i in tqdm(range(len(dataset))):  
                node_features = torch.FloatTensor(dataset[i, :, :, 0]).transpose(1,0) ## (time, n_node)
                edges = dense_to_sparse(torch.Tensor(adj_mx))[0]
                pyg_graph = Data(x=node_features, edge_index=edges, y=torch.tensor(labels[i]))
                g_all.append(pyg_graph)

        return g_all
        
    train_dataset = pyg_dataset(adj_mx, data['train_x'], data['train_y'], 'train')
    val_dataset = pyg_dataset(adj_mx, data['val_x'], data['val_y'], 'val')
    test_dataset = pyg_dataset(adj_mx, data['test_x'], data['test_y'], 'test')

    bs = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
