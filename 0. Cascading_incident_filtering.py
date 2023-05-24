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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils.convert import from_networkx



# Data Loading
data_root_path = '/media/usr/HDD/Data/NAVER'
partition_list = os.listdir(data_root_path)
partition_list = [p for p in partition_list if p[0]=='2']
partition_list = np.sort(partition_list)

data_path = '/media/usr/HDD/Working/Naver_Data/data_parsing'
data_extraction_path = '/media/usr/HDD/Data/NAVER_df'

sids_all = []
eventID_all = []

for partition in partition_list:
    try: 
        eventID_list = [filename.split('.')[0] for filename in os.listdir(os.path.join(data_path, 'networks', partition)) if filename[0] != '.']
        eventID_list = np.unique(eventID_list)
        eventID_all.append(eventID_list)

        for eventID in eventID_list:
            with open(os.path.join(data_path, 'networks', partition, '{}.pickle'.format(eventID)), 'rb') as f:
                accident_info = pickle.load(f)
            G = nx.read_gpickle(os.path.join(data_path, 'networks', partition, '{}.gpickle'.format(eventID)))

            sids_all.append(accident_info[1])
            sids_all.append(accident_info[2])
    except:
        continue

eventID_all = [x for y in eventID_all for x in y]
eventID_all = np.unique(eventID_all)
        
sids_all = [x for y in sids_all for x in y]
sids_all = np.unique(sids_all)

print('# of all Events, # of sids = ', len(eventID_all), len(sids_all))



filtered_ID = [eventID for eventID in eventID_all if eventID in os.listdir(data_extraction_path)]
print('filtered events : ', len(filtered_ID))


target_all = []
for eventID in tqdm(filtered_ID):
    try:
        with open('../Duration Estimation Thesis/feature_extraction/target/{}'.format(eventID), 'rb') as f:
            out = pickle.load(f)
    except:
        continue
        
    if out != None:
        out = [eventID] + out
        target_all.append(out)



target_all = pd.DataFrame(target_all, columns=['eventID', 'speed_drop', 'congestion_score', 'cascading_event', 'congestion_start_idx', 'congestion_duration'])
cascading_list = target_all[target_all.cascading_event == True]


accident_info_all = []
for eventID in filtered_ID:
    ## ===== Load extracted data =====
    eventID = str(eventID)
    
    with open(os.path.join(data_path, 'speeds', eventID, '{}.pickle'.format(eventID)), 'rb') as f:
        accident_info = pickle.load(f)
    G = nx.read_gpickle(os.path.join(data_path, 'speeds', eventID, '{}.gpickle'.format(eventID)))

    accident_sid = accident_info[0]['sids'][0]
    accident_created = accident_info[0]['created']
    
    if eventID not in list(cascading_list.eventID):
        continue

    accident_info_all.append(accident_info[0]) 

accident_all = pd.DataFrame(accident_info_all)
accident_all['accident_sid'] = accident_all['sids'].apply(lambda s: s[0])

accident_all.to_csv('data/accident_all.csv')