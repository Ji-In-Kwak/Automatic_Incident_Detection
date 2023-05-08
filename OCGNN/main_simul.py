import argparse
import os
# from dgl.data import register_data_args
import logging
import fire
from optim import trainer, TUtrainer, AEtrainer, Mytrainer
from optim.loss import loss_function,init_center
from datasets import dataloader_pyg as dataloader
# from datasets import TUloader_pyg as TUloader
from datasets import Myloader
from networks_pyg.init import init_model
import numpy as np
import torch
# from dgl import random as dr
import random

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


os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    set_seed(args.seed)

#     checkpoints_path=f'./checkpoints/{args.dataset}_{args.module}_{args.nu}+bestcheckpoint.pt'
#     logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
#     logger=logging.getLogger('OCGNN')

    checkpoints_path=f'./checkpoints/{args.exp_name}+bestcheckpoint.pt'
    logging.basicConfig(filename=f"./log/{args.exp_name}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    logger=logging.getLogger('OCGNN')
    print(args)


#     print('model:',args.module)
#     print('seed:',args.seed)

    if args.dataset in 'PROTEINS_full'+'ENZYMES'+'FRANKENSTEIN':
        train_loader, val_loader, test_loader, input_dim, label_dim=TUloader.loader(args)
        model=init_model(args,input_dim)
        model=TUtrainer.train(args,logger,train_loader,model,val_dataset=val_loader,path=checkpoints_path)
        # auc,ap,f1,acc,precision,recall,_= multi_graph_evaluate(args,checkpoints_path, 
        #     model, data_center,test_loader,radius,mode='test') 
        
        # torch.cuda.empty_cache()
        # print("Test AUROC {:.4f} | Test AUPRC {:.4f}".format(auc,ap))
        # print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        # logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
        # logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        # logger.info('\n')
    elif args.dataset in 'simulation':
        train_loader, val_loader, test_loader, input_dim = Myloader.simulation_loader(args)
        input_dim = 12
        model = init_model(args, input_dim)
        model = Mytrainer.train(args, logger, train_loader, val_loader, test_loader, model, checkpoints_path)

    else:
        data=dataloader.loader(args)
        model=init_model(args,data['input_dim'])
        if args.module != 'GAE':
            model=trainer.train(args,logger,data,model,checkpoints_path)
        else:
            model=AEtrainer.train(args,logger,data,model,checkpoints_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCGNN')
#     register_data_args(parser)
    parser.add_argument("--dataset", type=str, default='cora',
            help="dataset")
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability") 
    parser.add_argument("--nu", type=float, default=0.01, # 0.2 mu를 줄이면 r^2에 focus가 줄어듬
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
    parser.add_argument("--n-epochs", type=int, default=1000,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
            help="graph normalization (default=False)")
    parser.add_argument("--normalize", default='Standard',
            help='Normalization method in data preprocessing')
    parser.add_argument("--reverse", default=False,
            help='Reverse of the adjacency matrix')
    parser.add_argument("--exp-name", default='test',
            help='exp name to save model and log')
    parser.set_defaults(self_loop=True)
    parser.set_defaults(norm=False)
    args = parser.parse_args()
    if args.module=='GCN':
        #args.self_loop=True
        args.norm=True
    if args.module=='GAE':
        args.lr=0.002
        args.dropout=0.
        args.weight_decay=0.
        # args.n_hidden=32
    #     args.self_loop=True
    # if args.module=='GraphSAGE':
    #     args.self_loop=True

    if args.dataset in ('citeseer' + 'reddit'):
        args.normal_class=3
    if args.dataset in ('cora' + 'pubmed'):
        args.normal_class=2
    if args.dataset in 'TU_PROTEINS_full':
        args.normal_class=0

    fire.Fire(main(args))
