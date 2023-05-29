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


data_path = '/media/usr/HDD/Working/Naver_Data/data_parsing'
data_extraction_path = '/media/usr/HDD/Data/NAVER_df'


## load accident_all
accident_all = pd.read_csv('./data/accident_all.csv', index_col=0)
print("# of filtered Events = ", len(accident_all))

# Profile Extraction Functions
def profile_extraction2(speed_all):
    speed_all2 = speed_all.copy()
    # Day of Week => monday : 0, sunday : 6
    speed_all2['weekday'] = [s.weekday() for s in speed_all.index]
    speed_all2['timestamp'] = [s.time() for s in speed_all.index]
    
    profile_mean = speed_all2.groupby(['weekday', 'timestamp']).mean()
    profile_std = speed_all2.groupby(['weekday', 'timestamp']).std()
    
    speed_all2 = speed_all2.drop(['weekday', 'timestamp'], axis=1)
    
    return speed_all2, profile_mean, profile_std


target_sid = 1210005301 ## 1030001902    ## 1210005301
accident_case = accident_all[accident_all.loc[:, 'accident_sid'] == target_sid]
eventID = accident_case.eventId.iloc[0]
normalize = 'standard'

eventID = str(eventID)

# accident info : 0 : description / 1 : sid / 2 : sid 
# what sids?
with open(os.path.join(data_path, 'speeds', eventID, '{}.pickle'.format(eventID)), 'rb') as f:
    accident_info = pickle.load(f)
G = nx.read_gpickle(os.path.join(data_path, 'speeds', eventID, '{}.gpickle'.format(eventID)))

sid_list = accident_info[1] + accident_info[2]

accident_sid = accident_info[0]['sids'][0]
accident_created = accident_info[0]['created']

# feature extraction
with open(os.path.join(data_extraction_path, eventID), 'rb') as f:
    test = pickle.load(f)
speed_inflow = test['speed_inflow']
speed_outflow = test['speed_outflow']

speed_all = pd.concat([speed_inflow, speed_outflow], axis=1)
speed_all = speed_all.dropna(axis=1, how='all')

tmp = speed_all[accident_sid].iloc[:, 0].values
speed_all = speed_all.drop([accident_sid], axis=1)
speed_all[accident_sid] = tmp

## selected nodes
sid_list = list(set(list(speed_inflow.columns) + list(speed_outflow.columns) + [accident_sid]))
H = nx.subgraph(G, sid_list)

## speed_all 5min rolling & normalize
speed_all = speed_all.resample(rule='5T').mean()
if normalize == 'standard':
    scaler = StandardScaler() 
    arr_scaled = scaler.fit_transform(speed_all) 
    df_all_norm = pd.DataFrame(arr_scaled, columns=speed_all.columns,index=speed_all.index)
elif normalize == 'minmax':
    scaler = MinMaxScaler() 
    arr_scaled = scaler.fit_transform(speed_all) 
    df_all_norm = pd.DataFrame(arr_scaled, columns=speed_all.columns,index=speed_all.index)
elif normalize == 'profile':
    ## profile extraction
    # profile_all = profile_extraction(df_all_norm)
    speed_all, profile_mean, profile_std = profile_extraction2(speed_all)

    ## profile normalization
    date_index = np.arange(datetime(2020, 9, 2), datetime(2021, 3, 1), timedelta(days=1)).astype(datetime)
    df_all_norm = speed_all.copy()

    for date in date_index:
        date_index = np.arange(date, date+timedelta(days=1), timedelta(minutes=5)).astype(datetime)
        tmp = speed_all.loc[date_index]
        weekday = date.weekday()
        mean_tmp = profile_mean[288*weekday:288*(weekday+1)]
        std_tmp = profile_std[288*weekday:288*(weekday+1)]

        normalized = (tmp.values - mean_tmp) / std_tmp
        df_all_norm.loc[date_index] = normalized.values
        
        
############################################################################
# Recurrent Congestion
## smoothing
rolling_window = 6
speed_all = speed_all.rolling(rolling_window).mean()

## nan value
speed_all = speed_all.bfill(limit=36).ffill(limit=36).fillna(0, limit=288*10).dropna(axis=1)


ff_speed = speed_all.quantile(q=0.85, axis=0)
cong_speed = ff_speed * 0.6

congestion_label = pd.DataFrame(0, index=speed_all.index, columns=['RC'])
for i in range(len(speed_all)-12):
    ts = speed_all[i:i+12]
    n_cong_road = (ts.mean(axis=0) < cong_speed).sum()
    if n_cong_road > 0.5*speed_all.shape[1]:
        congestion_label.iloc[i+12] = 1
        
#############################################################################
# Nonrecurrent Congestion

## smoothing
rolling_window = 6
df_all_norm = df_all_norm.rolling(rolling_window).mean()

## nan value
df_all_norm = df_all_norm.bfill(limit=36).ffill(limit=36).fillna(0, limit=288*10).dropna(axis=1)



## Incident Data Plot
accident_case['created'] = pd.to_datetime(accident_case['created'])

for ix, row in accident_case.iterrows():
    t = row['created']
    accident_sid = row['accident_sid']
#     if (t.month != 1) and (t.month != 12):
#         continue
    if (t.month == 2):
        continue

    eventID = str(row['eventId'])
    with open(os.path.join(data_path, 'speeds', eventID, '{}.pickle'.format(eventID)), 'rb') as f:
        accident_info = pickle.load(f)
        
    accident_dt = accident_info[0]['created'].to_pydatetime()
    accident_dt = datetime.fromtimestamp(round(accident_dt.timestamp()/300)*300)
    accident_idx = np.where(df_all_norm.index == accident_dt)[0][0]
    df_pre = df_all_norm[accident_dt - timedelta(minutes=120):accident_dt]
    if df_pre.mean()[accident_sid] > 0:
        df_start_condition = (df_pre.mean() > df_all_norm)[accident_sid]
    else:
        df_start_condition = (df_pre.quantile(0.8) > df_all_norm)[accident_sid]
    df_start_condition = df_start_condition.astype(int).diff()==1
    
    congestion_start_list = np.where(df_start_condition==True)[0]
    congestion_start_list = np.unique([idx for idx in congestion_start_list if (accident_idx - 12*2 <= idx) & (idx <= accident_idx)])
    if len(congestion_start_list) == 0:
        congestion_start_time = accident_dt
    else:
        congestion_start_time = df_all_norm.index[congestion_start_list[0]].to_pydatetime()

#     df_end_condition = (df_pre.mean() < df_all_norm)[accident_sid]
#     congestion_end_list = np.where(df_end_condition==True)[0]
#     congestion_end_list = np.unique([idx for idx in congestion_end_list if (accident_idx - 12*2 <= idx) & (idx <= accident_idx)])
    
    df_avg_pre_new = df_all_norm[congestion_start_time - timedelta(minutes=60):accident_dt]
    zero_condition = pd.DataFrame(0, index=df_avg_pre_new.index, columns=df_avg_pre_new.columns)
    df_end_condition = (np.maximum(zero_condition.mean(), df_avg_pre_new.mean()) < df_all_norm)[accident_sid]
    df_end_condition = df_end_condition.astype(int).diff()==1
    congestion_end_list = np.where(df_end_condition==True)[0]
    congestion_end_list = [idx for idx in congestion_end_list if (accident_idx <= idx) & (idx <= accident_idx + 12*4)]
    if len(congestion_end_list) == 0:
        print("!!!")
        congestion_end_list = [accident_idx+12*4]
    congestion_end_time = df_all_norm.index[congestion_end_list[0]].to_pydatetime()

#     congestion_start_time, congestion_end_time, _ = get_congestion_start_end(df_avg, accident_info, std_min=30, start_period=12*2, end_period=12*24, gap=120/5)

    ## Label generation
    congestion_label.loc[congestion_start_time:congestion_end_time, 'NRC'] = 1


#############################################################################
# Final Label
train_df = df_all_norm[df_all_norm.index < datetime(2020, 12, 1)]
val_df = df_all_norm[(df_all_norm.index >= datetime(2020, 12, 1)) & (df_all_norm.index < datetime(2021, 1, 1))]
test_df = df_all_norm[(df_all_norm.index >= datetime(2021, 1, 1)) & (df_all_norm.index < datetime(2021, 2, 1))]

train_label = congestion_label[congestion_label.index < datetime(2020, 12, 1)]
val_label = congestion_label[(congestion_label.index >= datetime(2020, 12, 1)) & (congestion_label.index < datetime(2021, 1, 1))]
test_label = congestion_label[(congestion_label.index >= datetime(2021, 1, 1)) & (congestion_label.index < datetime(2021, 2, 1))]

def get_x_y(speed_df, label_df):
    X_all = []
    y_all = []
    for i in range(len(speed_df)-12):
        if label_df['NRC'][i+12] == 1:
            X_all.append(speed_df.iloc[i:i+12, :])
            y_all.append(1)
        elif label_df['RC'][i+12] == 1:
            X_all.append(speed_df.iloc[i:i+12, :])
            y_all.append(0)
    X_all = np.stack(X_all)
    y_all = np.stack(y_all)
    
    return X_all, y_all

train_X, train_y = get_x_y(train_df, train_label)
val_X, val_y = get_x_y(val_df, val_label)
test_X, test_y = get_x_y(test_df, test_label)


dataset = '{}_mtsc'.format(target_sid)
os.makedirs('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}'.format(dataset), exist_ok=True)

np.savez('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/train.npz'.format(dataset), x=train_X, y=train_y)
np.savez('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/val.npz'.format(dataset), x=val_X, y=val_y)
np.savez('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/test.npz'.format(dataset), x=test_X, y=test_y)


## network graph
sid_list = list(map(int, train_df.columns))
H = nx.subgraph(H, sid_list)
nx.write_gpickle(H.copy(), "/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/sensor_graph.gpickle".format(dataset))