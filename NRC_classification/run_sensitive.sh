#!/bin/bash



# sensetivity analysis
dataset='1210005301_mtsc'
pooling=sum
self_loop=True
nu=0.01
for hidden_dim in 8 16 32 64 128
do
    # python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_GCN_${pooling}_${nu}_${hidden_dim}
    python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_STGCN_${pooling}_bias_${nu}_${hidden_dim}
done

nu=0.05
for hidden_dim in 8 16 32 64 128
do
    # python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_GCN_${pooling}_${nu}_${hidden_dim}
    python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_STGCN_${pooling}_bias_${nu}_${hidden_dim}
done

nu=0.10
for hidden_dim in 8 16 32 64 128
do
    # python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_GCN_${pooling}_${nu}_${hidden_dim}
    python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_STGCN_${pooling}_bias_${nu}_${hidden_dim}
done

nu=0.20
for hidden_dim in 8 16 32 64 128
do
    # python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_GCN_${pooling}_${nu}_${hidden_dim}
    python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_STGCN_${pooling}_bias_${nu}_${hidden_dim}
done

nu=0.30
for hidden_dim in 8 16 32 64 128
do
    # python main_SAD.py --dataset ${dataset} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_GCN_${pooling}_${nu}_${hidden_dim}
    python main_SAD.py --dataset ${dataset} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --n-hidden ${hidden_dim} --exp-name ${dataset}_STGCN_${pooling}_bias_${nu}_${hidden_dim}
done