import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import argparse
import logging
from datetime import datetime, timedelta, date
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils.convert import from_networkx


# from networks_pyg.GCN import *
from datasets.Myloader import traffic_mtsc_loader
from networks_pyg.init import init_model
from optim.loss_my import anomaly_score


parser = argparse.ArgumentParser(description='OCGNN')
parser.add_argument("--dataset", type=str, default='cora',
        help="dataset")
parser.add_argument("--kfold", type=int, default=None,
        help="K-fold cross validation")
parser.add_argument("--dropout", type=float, default=0.25,
        help="dropout probability")
parser.add_argument("--nu", type=float, default=0.01, # 0.2
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
parser.add_argument("--n-epochs", type=int, default=100,
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
parser.add_argument("--reverse", default=False,
        help='Reverse of the adjacency matrix')
parser.add_argument("--pooling", default='sum',
        help='Type of pooling layer to aggregate embeddings into graph embedidng')
parser.add_argument("--exp-name", default='test',
        help='exp name to save model and log')
parser.set_defaults(self_loop=True)
parser.set_defaults(norm=False)
args = parser.parse_args()



## Data Loading
data_root_path = '/media/usr/HDD/Data/NAVER'
partition_list = os.listdir(data_root_path)
partition_list = [p for p in partition_list if p[0]=='2']
partition_list = np.sort(partition_list)

data_path = '/media/usr/HDD/Working/Naver_Data/data_parsing'


## https://github.com/mangushev/mtad-tf/blob/main/evaluate.py
#just like onmianomaly, no delta. If we hit anuthing in anomaly interval, whole anomaly segment is correctly identified
#-----------------------
#1|0|1|1|1|0|0|0|1|1|1|1  Labels
#-----------------------
#0|0|0|1|1|0|0|0|0|0|1|0  Predictions
#-----------------------
#0|0|1|1|1|0|0|0|1|1|1|1  Adjusted
#-----------------------
def adjust_predictions(predictions, labels):
    adjustment_started = False
    new_pred = predictions

    for i in range(len(predictions)):
        if labels[i] == 1:
            if predictions[i] == 1:
                if not adjustment_started:
                    adjustment_started = True
                    for j in range(i, 0, -1):
                        if labels[j] == 1:
                            new_pred[j] = 1
                        else:
                            break
    else:
        adjustment_started = False

    if adjustment_started:
        new_pred[i] = 1
      
    return new_pred

def evaluate(true, pred, score, adjust = False, plot=False, print_=False):
    if adjust:
        pred = adjust_predictions(pred, true)
    CM = confusion_matrix(true, pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    acc = accuracy_score(true, pred)
    # auc = roc_auc_score(true, pred)
    auc = roc_auc_score(true, score)
#     far = FP / (FP+TN)
    far = FP / (TP+FP)
    pre = precision_score(true, pred, pos_label=1)
    rec = recall_score(true, pred, pos_label=1)
    macro_f1 = f1_score(true, pred, average='macro')
    weighted_f1 = f1_score(true, pred, average='weighted')
    ap = average_precision_score(true, score)
    # ap = average_precision_score(true, pred)
    if plot:
        plt.figure(figsize=(40, 5))
        plt.plot(true)
        plt.plot(pred)
    if print_:
        print('Accuracy \t{:.4f}'.format(acc))
        print('AUC score \t{:.4f}'.format(auc))
        print('FAR score \t{:.4f}'.format(far))
        print('Precision \t{:.4f}'.format(pre))
        print('Recall   \t{:.4f}'.format(rec))
        print('Macro F1 \t{:.4f}'.format(macro_f1))
        print('weighted F1 \t{:.4f}'.format(weighted_f1))
        print('Avg Precision \t{:.4f}'.format(ap))
        print(classification_report(true, pred))
    return [acc, auc, far, pre, rec, macro_f1, weighted_f1, ap]



## All results
## load accident_all
accident_all = pd.read_csv('../data/accident_all.csv', index_col=0)
accident_all['created'] = pd.to_datetime(accident_all['created'])
print("# of filtered Events = ", len(accident_all))

result_all = []
target_sid = int(args.dataset.split('_')[0])
print(args.dataset, target_sid)
# target_sid = 1030001902 ## 1210005301  ## 1030001902  ## 1220005401  ## 1210003000  ## 1130052300
accident_case = accident_all[accident_all.loc[:, 'accident_sid'] == target_sid]
eventID = accident_case.eventId.iloc[0]


if args.gpu < 0:
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{args.gpu}')

for normalize in ['standard']:
    # data
#     args.dataset = '{}_CV2'.format(target_sid)
    args.normalize = normalize
    args.bias = True
    args.bidirect = False

    for k in range(10):
        args.kfold = k
        train_loader, val_loader, test_loader = traffic_mtsc_loader(args)
        print("##############################")

        for nu in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for name, module in zip(['GCN', 'GAT', 'GraphSAGE', 'STGCN',], ['GCN_gc','GAT_gc', 'GraphSAGE_gc', 'STGCN']):
                args.module=module
                args.nu = nu
                args.exp_name = f'{args.dataset}_{args.kfold}_{name}_{args.pooling}_{args.nu}_{args.self_loop}'


                checkpoints_path=f'./checkpoints_SAD_CV/{args.dataset}/{args.exp_name}+bestcheckpoint.pt'
                print(checkpoints_path)

                # model
                input_dim = 24
                model = init_model(args, input_dim)
                model.load_state_dict(torch.load(checkpoints_path, map_location=device)['model'])
                model.to(device=device)
                model.eval()
                data_center = torch.load(checkpoints_path, map_location=device)['data_center']
                radius = torch.load(checkpoints_path, map_location=device)['radius'].to(device=device)

                out_all, dist_all, score_all = [], [], []
                for ix, data in (enumerate(val_loader)):
                    output = model(data.to(device=device))
                    out_all.append(output.cpu().detach().numpy())
                    dist, _ = anomaly_score(data_center, output, radius)
                    dist_all.append(dist.cpu().detach().numpy())
                dist_all = np.concatenate(dist_all)
#                 new_radius = np.quantile(np.sqrt(dist_all), 1 - args.nu)
                new_radius = np.quantile(np.sqrt(dist_all), 0.97)
                print(radius.data, new_radius)


                out_all, dist_all, score_all = [], [], []
                label_all = []
                for ix, data in (enumerate(test_loader)):
                    output = model(data.to(device=device))
                    out_all.append(output.cpu().detach().numpy())
                    label_all.append(data.y.cpu().detach().numpy())
                    dist, score = anomaly_score(data_center, output, new_radius) # new_radius
                    dist_all.append(dist.cpu().detach().numpy())
                    score_all.append(score.cpu().detach().numpy())
                label_all = np.concatenate(label_all)
                score_all = np.concatenate(score_all)
                dist_all = np.concatenate(dist_all)

                pred = (score_all > 0).astype(int)

                acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(label_all, pred, score_all, adjust=False, plot=False)
                result_all.append([name, k, args.nu, False, rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])
                acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(label_all, pred, score_all, adjust=True, plot=False)
                result_all.append([name, k, args.nu, True, rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])
                
                
result_all = pd.DataFrame(result_all, columns=['model', 'Kfold', 'nu', 'adjust', 'DR', 'far', 'precision', 'recall', 'acc', 'AUC', 'F1_macro', 'F1_weight', 'AP'])
result_all.to_csv('result/{}_sumpool_0.97.csv'.format(args.dataset))          