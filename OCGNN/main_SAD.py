#-*- coding:utf-8 -*-

import os
import sys
import time
import random
import math
import pickle
import unicodedata
import argparse
import fire

# import ast as ast
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime, timedelta, date
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils.convert import from_networkx


from networks_pyg.GCN import *
from datasets.Myloader import traffic_loader
from optim import DeepSAD_trainer
from optim.loss import loss_function,init_center
from datasets import dataloader_pyg as dataloader
from datasets import Myloader
from networks_pyg.init import init_model


# Set random seed
def set_seed(SEED=0):
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import warnings
warnings.filterwarnings('ignore')


# # Data Loading
# data_root_path = '/media/usr/HDD/Data/NAVER'
# partition_list = os.listdir(data_root_path)
# partition_list = [p for p in partition_list if p[0]=='2']
# partition_list = np.sort(partition_list)

# data_path = '/media/usr/HDD/Working/Naver_Data/data_parsing'

# sids_all = []
# eventID_all = []

# for partition in partition_list:
#     try: 
#         eventID_list = [filename.split('.')[0] for filename in os.listdir(os.path.join(data_path, 'networks', partition)) if filename[0] != '.']
#         eventID_list = np.unique(eventID_list)
#         eventID_all.append(eventID_list)

#         for eventID in eventID_list:
#             with open(os.path.join(data_path, 'networks', partition, '{}.pickle'.format(eventID)), 'rb') as f:
#                 accident_info = pickle.load(f)
#             G = nx.read_gpickle(os.path.join(data_path, 'networks', partition, '{}.gpickle'.format(eventID)))

#             sids_all.append(accident_info[1])
#             sids_all.append(accident_info[2])
#     except:
#         continue

# eventID_all = [x for y in eventID_all for x in y]
# eventID_all = np.unique(eventID_all)
        
# sids_all = [x for y in sids_all for x in y]
# sids_all = np.unique(sids_all)

# print('# of all Events, # of sids = ', len(eventID_all), len(sids_all))

# data_extraction_path = '/media/usr/HDD/Data/NAVER_df'
# filtered_ID = [eventID for eventID in eventID_all if eventID in os.listdir(data_extraction_path)]

## load accident_all
accident_all = pd.read_csv('../data/accident_all.csv', index_col=0)
print("# of filtered Events = ", len(accident_all))


def main(args, target_sid, eventID):
    set_seed(args.seed)
  
    checkpoints_path=f'./checkpoints_SAD_again/{args.exp_name}+bestcheckpoint.pt'
    logging.basicConfig(filename=f"./log_SAD_again/{args.exp_name}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    logger=logging.getLogger('DeepSAD')

    eventID = str(eventID)


    # DataLoader
    train_loader, val_loader, test_loader = traffic_loader(args, target_sid, method = 'semisup')  ###### incident label도 포함하게 수정
    print(len(train_loader), len(val_loader), len(test_loader))

    # Model Train
    input_dim = 24
    print(args)
    model = init_model(args, input_dim).to(device=f'cuda:{args.gpu}')
    model = DeepSAD_trainer.train(args, logger, train_loader, val_loader, test_loader, model, checkpoints_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSAD')
#     register_data_args(parser)
    parser.add_argument("--dataset", type=str, default='cora',
            help="dataset")
    parser.add_argument("--dropout", type=float, default=0.25,
            help="dropout probability") 
    parser.add_argument("--nu", type=float, default=0.1, # 0.2 mu를 줄이면 r^2에 focus가 줄어듬
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--seed", type=int, default=52,
            help="random seed, -1 means dont fix seed")
    parser.add_argument("--module", type=str, default='GraphSAGE',
            help="GCN/GAT/GIN/GraphSAGE/GAE")
    parser.add_argument('--n-worker', type=int,default=1,
            help='number of workers when dataloading')
    parser.add_argument('--batch-size', type=int,default=128,
            help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--normal-class", type=int, default=0,
            help="normal class")
    parser.add_argument("--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", type=bool, default=False,
            help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
            help="graph normalization (default=False)")
    parser.add_argument("--normalize", default='Standard',
            help='Normalization method in data preprocessing')
    parser.add_argument("--reverse", default=False,
            help='Reverse of the adjacency matrix')
    parser.add_argument("--pooling", default='sum',
            help='Type of pooling layer to aggregate embeddings into graph embedidng')
    parser.add_argument("--exp-name", default='test',
            help='exp name to save model and log')
    parser.set_defaults(norm=False)
    args = parser.parse_args()
#     if args.module=='GCN_gc':
#         args.self_loop=True
#         args.norm=True
#     if args.module=='GraphSAGE_gc':
#         args.self_loop=False
#     if args.module=='STGCN':
#         args.self_loop=True
#     if args.module=='GAE':
#         args.lr=0.002
#         args.dropout=0.
#         args.weight_decay=0.
        # args.n_hidden=32
    #     args.self_loop=True
    # if args.module=='GraphSAGE':
    #     args.self_loop=True

    # if args.dataset in ('citeseer' + 'reddit'):
    #     args.normal_class=3
    # if args.dataset in ('cora' + 'pubmed'):
    #     args.normal_class=2
    # if args.dataset in 'TU_PROTEINS_full':
    #     args.normal_class=0


    target_sid = 1210005301   ## 1030001902   ## 1210005301
    accident_case = accident_all[accident_all.loc[:, 'accident_sid'] == target_sid]
    eventID = accident_case.eventId.iloc[0]

    fire.Fire(main(args, target_sid, eventID))