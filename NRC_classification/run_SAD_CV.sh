#!/bin/bash


## sum pool
dataset='1210005301_CV'    ## 1210005301  ## 1030001902  ## 1220005401  ## 1210003000  ## 1130052300
pooling=sum
self_loop=True

for k in 0 1 2 3 4 5 6 7 8 9
do
    echo $k
    for nu in 0.01 0.03 0.05 0.1 0.2 0.3
    do
        python main_SAD_CV.py --dataset ${dataset} --kfold ${k} --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_${k}_GCN_${pooling}_${nu}_${self_loop}
        python main_SAD_CV.py --dataset ${dataset} --kfold ${k} --normalize standard --module GAT_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_${k}_GAT_${pooling}_${nu}_${self_loop} 
        python main_SAD_CV.py --dataset ${dataset} --kfold ${k} --normalize standard --module GraphSAGE_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_${k}_GraphSAGE_${pooling}_${nu}_${self_loop} 
        python main_SAD_CV.py --dataset ${dataset} --kfold ${k} --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu ${nu} --n-epochs 200 --exp-name ${dataset}_${k}_STGCN_${pooling}_${nu}_${self_loop}
    done
done


python detection_results.py --dataset ${dataset} --self-loop ${self_loop} --pooling ${pooling}