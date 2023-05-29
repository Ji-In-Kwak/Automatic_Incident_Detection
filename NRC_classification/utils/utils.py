#-*- coding:utf-8 -*-

import os
import sys
import time
import random
import math
import pickle
import unicodedata
import ast

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from datetime import datetime, timedelta
# from dtaidistance import dtw
from multiprocessing import Pool



def get_congestion_start_end(df_avg, accident_info, std_min=30, start_period=120, end_period=240, gap=120):
    '''
    For Sensitivity, we change the 'std_min' from 1 to 60 (Benchmark 30)
    '''
    accident_dt = accident_info[0]['created'].to_pydatetime()
    accident_dt = datetime.fromtimestamp(round(accident_dt.timestamp( ) / 300 ) *300)
    accident_idx = np.where(df_avg.index == accident_dt)[0][0]

    df_ema_05 = df_avg.ewm(span=1).mean()
    df_ema_30 = df_avg.ewm(span=6).mean()
    df_avg_pre = df_avg[accident_dt - timedelta(minutes=start_period*5):accident_dt]

    # ===== Find congestion start idx =====
    ## 1. thresholds
    df_speed = (df_ema_05 <= df_avg_pre.mean())

    ## 2. moving average
    df_trend = (df_ema_05 <= df_ema_30)

    ## 3. volatility breakthrough
    df_range = df_avg.rolling(std_min).std(min_periods=0)
    df_range = df_range.fillna(0)

    df_threshold = df_avg - df_range
    df_threshold = df_threshold.shift(1).fillna(0)
    df_vola = df_avg <= df_threshold

    ## select false->true idx
    df_start_condition = df_speed & df_trend & df_vola
    df_start_condition = df_start_condition.astype(int).diff()==1

    congestion_start_list = np.where(df_start_condition==True)[0]
    congestion_start_list = [idx for idx in congestion_start_list if (accident_idx - start_period <= idx) & (idx <= accident_idx)]

    if len(congestion_start_list) == 0:
        congestion_start_idx = accident_idx
    else:
        congestion_start_idx = congestion_start_list[0]
    congestion_start_time = df_avg.index[congestion_start_idx].to_pydatetime()

    # ===== Find congestion end idx =====
    # congestion end datetime
    df_avg_pre_new = df_avg[congestion_start_time - timedelta(minutes=start_period*5):congestion_start_time]
    df_end_condition = df_avg_pre_new.mean() < df_avg.rolling(std_min).mean(min_periods=0)
    df_end_condition = df_end_condition.astype(int).diff()==1

    ## select false->true idx
    congestion_end_list = np.where(df_end_condition==True)[0]
    congestion_end_list = [idx for idx in congestion_end_list if (congestion_start_idx <= idx) & (idx <= accident_idx + end_period)]

    if len(congestion_end_list) == 0:
        if df_avg_pre_new.mean() <= df_avg[accident_dt]:
            congestion_end_idx = accident_idx
        else:
            congestion_end_idx = accident_idx+gap # domain knowledge
    else:
        congestion_end_idx = congestion_end_list[0]
    congestion_end_time = df_avg.index[congestion_end_idx].to_pydatetime()

    # ===== Find duration =====
    duration = congestion_end_time - congestion_start_time
    duration = duration.seconds//60

    return congestion_start_time, congestion_end_time, duration