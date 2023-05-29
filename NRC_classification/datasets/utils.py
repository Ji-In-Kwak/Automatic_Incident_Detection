from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
#import cPickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, download_url
import pdb
import argparse

class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


def load_new_data(dataset):

    print('start loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    print(os.getcwd())
    with open('../dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if attr is not None:
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feat_dict = {}
                for k in range(g.number_of_nodes()):
                    node_feat_dict[k] = np.array(node_features[k])
                nx.set_node_attributes(g, node_feat_dict, name="x")
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False
                
            ## node tag encoding 필요!!!

            #assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            pyg_graph = from_networkx(g)
            pyg_graph.x = pyg_graph.x.type(torch.FloatTensor)
            pyg_graph.y = torch.tensor([l])
#             g_list.append(GNNGraph(g, l, node_tags, node_features))
            g_list.append(pyg_graph)


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(feat_dict))

    # info = {'num_classes':len(label_dict), 'num_node_tag':len(feat_dict), 'num_node_feat':node_features.shape[1]}
    info = [len(label_dict), len(feat_dict), node_features.shape[1]]

    return g_list, info