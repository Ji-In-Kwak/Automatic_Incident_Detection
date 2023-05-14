#!/bin/bash

# ## profile
# for nu in 0.01 0.05 0.1 0.2
# do
#     python main_traffic.py --dataset Accident_profile --normalize profile --module GCN_gc --reverse False --nu $nu --n-epochs 200 --exp-name profile_all_GCN_sum_${nu}_1902
#     python main_traffic.py --dataset Accident_profile --normalize profile --module GCN_gc --reverse True --nu $nu --n-epochs 200 --exp-name profile_all_GCN_rev_sum_${nu}_1902
#     python main_traffic.py --dataset Accident_profile --normalize profile --module GraphSAGE_gc --reverse False  --nu $nu --n-epochs 200 --exp-name profile_all_GraphSAGE_sum_${nu}_1902
#     python main_traffic.py --dataset Accident_profile --normalize profile --module GraphSAGE_gc --reverse True --nu $nu --n-epochs 200 --exp-name profile_all_GraphSAGE_rev_sum_${nu}_1902
# done

# ## standard
# for nu in 0.01 0.05 0.1 0.2
# do
#     python main_traffic.py --dataset Accident_standard --normalize standard --module GCN_gc --reverse False --nu $nu --n-epochs 200 --exp-name standard_GCN_sum_${nu}_sl_5301 
#     python main_traffic.py --dataset Accident_standard --normalize standard --module GCN_gc --reverse True --nu $nu --n-epochs 200 --exp-name standard_GCN_rev_sum_${nu}_sl_5301 
#     python main_traffic.py --dataset Accident_standard --normalize standard --module GraphSAGE_gc --reverse False  --nu $nu --n-epochs 200 --exp-name standard_GraphSAGE_sum_${nu}_sl_5301 
#     python main_traffic.py --dataset Accident_standard --normalize standard --module GraphSAGE_gc --reverse True --nu $nu --n-epochs 200 --exp-name standard_GraphSAGE_rev_sum_${nu}_sl_5301 
#     python main_traffic.py --dataset Accident_standard --normalize standard --module STGCN --reverse False --nu $nu --n-epochs 200 --exp-name standard_STGCN_sum_${nu}_sl_5301
# done

## sum pool
pooling=sum
self_loop=True
for nu in 0.01 0.05 0.1 0.2
do
    python main_SAD.py --dataset Accident_standard --normalize standard --module GCN_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_GCN_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset Accident_standard --normalize standard --module GCN_gc --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_GCN_rev_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset Accident_standard --normalize standard --module GraphSAGE_gc --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_GraphSAGE_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset Accident_standard --normalize standard --module GraphSAGE_gc --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_GraphSAGE_rev_${pooling}_${nu}_${self_loop}_5301 
    python main_SAD.py --dataset Accident_standard --normalize standard --module STGCN --reverse False --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --exp-name standard_STGCN_${pooling}_${nu}_${self_loop}_5301
    python main_SAD.py --dataset Accident_standard --normalize standard --module STGCN --reverse True --self-loop ${self_loop} --pooling ${pooling} --nu $nu --n-epochs 200 --pooling $pooling --exp-name standard_STGCN_rev_${pooling}_${nu}_${self_loop}_5301
done



## Simulation

# for nu in 0.05 0.1 0.2
# do
#     python main_simul.py --dataset simulation --normalize standard --module GCN_gc --nu $nu --exp-name simulation_GCN_sum_$nu
#     python main_simul.py --dataset simulation --normalize standard --module GCN_traffic --nu $nu --exp-name simulation_GCN_rev_sum_$nu
#     python main_simul.py --dataset simulation --normalize standard --module GraphSAGE_gc --nu $nu --exp-name simulation_GraphSAGE_sum_$nu
# done