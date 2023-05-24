#!/bin/bash


## sum pool
dataset='1210005301'
pooling=sum
self_loop=True
for nu in 0.01 0.05 0.1
do
    python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_GCN_${pooling}_${nu}_${self_loop}
#     python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_GCN_rev_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset ${dataset} --normalize standard --module GAT_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_GAT_${pooling}_${nu}_${self_loop} 
#     python main_SAD.py --dataset ${dataset} --normalize standard --module GAT_gc --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name standard_GCN_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset ${dataset} --normalize standard --module GraphSAGE_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_GraphSAGE_${pooling}_${nu}_${self_loop} 
#     python main_SAD.py --dataset ${dataset} --normalize standard --module GraphSAGE_gc --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_GraphSAGE_rev_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_STGCN_${pooling}_${nu}_${self_loop}
#     python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --pooling $pooling --exp-name standard_STGCN_rev_${pooling}_${nu}_${self_loop}_5301
done



# ## Simulation
# dataset=5301_simulation
# pooling=sum
# self_loop=True
# for nu in 0.01 0.05 0.1 0.2
# do
#     python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 100 --exp-name ${dataset}_GCN2_${pooling}_${nu}_${self_loop}
#     # python main_SAD.py --dataset ${dataset} --normalize standard --module GAT_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 100 --exp-name ${dataset}_GAT_${pooling}_${nu}_${self_loop}
#     # python main_SAD.py --dataset ${dataset} --normalize standard --module GraphSAGE_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 100 --exp-name ${dataset}_GraphSAGE_${pooling}_${nu}_${self_loop}
#     # python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 100 --exp-name ${dataset}_STGCN_${pooling}_${nu}_${self_loop}
# done