import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm
from sklearn.metrics import *
import torch
import torch.nn.functional as F


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## load accident_all
accident_all = pd.read_csv('../data/accident_all.csv', index_col=0)
print("# of filtered Events = ", len(accident_all))


# # Profile Extraction Functions
# def profile_extraction2(speed_all):
#     # Day of Week => monday : 0, sunday : 6
#     speed_all['weekday'] = [s.weekday() for s in speed_all.index]
#     speed_all['timestamp'] = [s.time() for s in speed_all.index]
    
#     profile_mean = speed_all.groupby(['weekday', 'timestamp']).mean()
#     profile_std = speed_all.groupby(['weekday', 'timestamp']).std()
    
#     speed_all = speed_all.drop(['weekday', 'timestamp'], axis=1)
    
#     return speed_all, profile_mean, profile_std



target_sid = 1210005301  ## 1210005301    ## 1030001902
accident_case = accident_all[accident_all.loc[:, 'accident_sid'] == target_sid]
eventID = accident_case.eventId.iloc[0]
normalize = 'standard'


## Data Loading
print('Data Loading ....')
target_sid = 1210005301
dataset = '{}_mtsc'.format(target_sid)

train = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/train.npz'.format(dataset))
val = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/val.npz'.format(dataset))
test = np.load('/media/usr/SSD/jiin/naver/Automatic_Incident_Detection/data/{}/test.npz'.format(dataset))

H = nx.read_gpickle("../data/{}/sensor_graph.gpickle".format(dataset))


## Evaluation
# def adjust_predictions(predictions, labels):
#     adjustment_started = False
#     new_pred = predictions

#     for i in range(len(predictions)):
#         if (labels[i] == 1) & (predictions[i] == 1):
#             if not adjustment_started:
#                 adjustment_started = True
#                 for j in range(i, 0, -1):
#                     if labels[j] == 1:
#                         new_pred[j] = 1
#                     else:
#                         break
#         else:
#             adjustment_started = False

#         if adjustment_started:
#             new_pred[i] = 1

#     return new_pred


def evaluate(true, pred, score, adjust = False, plot=False, print_=False):
#     true = label_all
#     pred = list(map(int, [s>0 for s in score_all]))
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


result_all = []


##################################
## Univariate ###
##################################
# result_all = []

# # SND algorithm

# # df_all_norm, _, _ = preprocessing(eventID, 'profile')


# # Supervised
# ## Data
# train_data, test_data = [], []
# train_label_cls, test_label_cls = [], []

# for i in tqdm(range(train_df.shape[0]-24)):
#     train_data.append(train_df.iloc[i:i+24][target_sid].values)
#     if 1 in train_label.iloc[i:i+24]['label'].values:
#         train_label_cls.append(1)
#     else:
#         train_label_cls.append(0) ## make train data into vector format with reshape(-1)
# for i in tqdm(range(test_df.shape[0]-24)):
#     test_data.append(test_df.iloc[i:i+24][target_sid].values)
#     test_label_cls.append(test_label.iloc[i+24].values)
# train_data = np.stack(train_data)
# test_data = np.stack(test_data)


# ### SVM
# from sklearn.svm import SVC

# clf = SVC()
# clf.fit(train_data, train_label_cls)

# true = test_label_cls
# pred = clf.predict(test_data)

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['univariate', 'SVM', rec, far, pre, rec, acc, auc, macro_f1, ap])


# ### MLP
# from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(hidden_layer_sizes=(32, 32))
# clf.fit(train_data, train_label_cls)

# true = test_label_cls
# pred = clf.predict(test_data)

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['univariate', 'ANN', rec, far, pre, rec, acc, auc, macro_f1, ap])




# # Unsupervised
# ## Data
# train_data, test_data = [], []
# for i in tqdm(range(train_df.shape[0]-24)):
#     if 1 in train_label.iloc[i:i+24]['label'].values:
#         continue
#     else:
#         train_data.append(train_df.iloc[i:i+24][target_sid])
# for i in tqdm(range(test_df.shape[0]-24)):
#     test_data.append(test_df.iloc[i:i+24][target_sid])
# train_data = np.stack(train_data)
# test_data = np.stack(test_data)



# ### OCSVM
# from sklearn.svm import OneClassSVM

# clf = OneClassSVM(kernel='rbf').fit(train_data[:])

# true = test_label.iloc[24:].label.values
# pred = clf.predict(test_data)
# pred = [1 if p==-1 else 0 for p in pred]

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['univariate', 'OCSVM', rec, far, pre, rec, acc, auc, macro_f1, ap])


# ### LOF
# from sklearn.neighbors import LocalOutlierFactor

# clf = LocalOutlierFactor(n_neighbors=2, novelty=True)
# clf.fit(train_data)

# true = test_label.iloc[24:].label.values
# pred = clf.predict(test_data)
# pred = [1 if p==-1 else 0 for p in pred]

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['univariate', 'LOF', rec, far, pre, rec, acc, auc, macro_f1, ap])


# ### IF
# from sklearn.ensemble import IsolationForest

# clf = IsolationForest(random_state=0)
# clf.fit(train_data)

# true = test_label.iloc[24:].label.values
# pred = clf.predict(test_data)
# pred = [1 if p==-1 else 0 for p in pred]

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['univariate', 'IForest', rec, far, pre, rec, acc, auc, macro_f1, ap])




##################################
## Multivariate ###
##################################

# Supervised
## Data
# from sklearn.decomposition import PCA

# train_data, test_data = [], []
# train_label_cls, test_label_cls = [], []

# for i in tqdm(range(train_df.shape[0]-24)):
#     train_data.append(train_df.iloc[i:i+24].values.reshape(-1))
#     if 1 in train_label.iloc[i:i+24]['label'].values:
#         train_label_cls.append(1)
#     else:
#         train_label_cls.append(0) ## make train data into vector format with reshape(-1)
# for i in tqdm(range(test_df.shape[0]-24)):
#     test_data.append(test_df.iloc[i:i+24].values.reshape(-1))
#     test_label_cls.append(test_label.iloc[i+24].values)
# train_data = np.stack(train_data)
# test_data = np.stack(test_data)

# pca_1d = PCA(n_components=32)
# train_data_1d = pca_1d.fit_transform(train_data)
# test_data_1d = pca_1d.transform(test_data)
# train_data_1d.shape, test_data_1d.shape


# DATA
train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
test_data = test['x'].reshape(test['x'].shape[0], -1)

train_label = train['y']
test_label = test['y']


# SVM
print('Multivariate SVM')
from sklearn.svm import SVC

clf = SVC(probability=True)
clf.fit(train_data, train_label)

true = test_label
pred = clf.predict(test_data)
score = clf.predict_proba(test_data)[:, 1]

acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
result_all.append(['multivariate', 'SVM', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


# MLP
print('Multivariate MLP')
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(64, 64))
clf.fit(train_data, train_label)

true = test_label
pred = clf.predict(test_data)
score = clf.predict_proba(test_data)[:, 1]

acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
result_all.append(['multivariate', 'MLP', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])



# ## OCSVM
# print('Multivariate OCSVM')
# from sklearn.svm import OneClassSVM

# clf = OneClassSVM(kernel='rbf').fit(train_data)

# true = test_label
# pred = clf.predict(test_data)
# pred = [1 if p == -1 else 0 for p in pred]
# score = clf.score_samples(test_data)
# score = score 

# acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
# result_all.append(['multivariate', 'OCSVM', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])




## multivariate 3D data
train_data = np.transpose(train['x'], (0, 2, 1))  ## (n_samples, n_node, seq_len)
test_data = np.transpose(test['x'], (0, 2, 1))

train_label = train['y']
test_label = test['y']


# CNN
print('Multivariate CNN')
from sktime.classification.deep_learning import CNNClassifier

clf = CNNClassifier(n_epochs=100, batch_size=64, random_state=0)
clf.fit(train_data, train_label)

true = test_label
pred = clf.predict(test_data)
score = clf.predict_proba(test_data)[:, 1]

acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
result_all.append(['multivariate', 'CNN', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


# LSTM-FCN
print('Multivariate LSTM-FCN')
from sktime.classification.deep_learning import LSTMFCNClassifier

clf = LSTMFCNClassifier(n_epochs=100, batch_size=64, random_state=0, verbose=True)
clf.fit(train_data, train_label)

true = test_label
pred = clf.predict(test_data)
score = clf.predict_proba(test_data)[:, 1]

acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
result_all.append(['multivariate', 'LSTM-FCN', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


# DeepSAD
print('Raw DeepSAD')

# DATA
train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
test_data = test['x'].reshape(test['x'].shape[0], -1)

train_label = np.where(train['y']==1, -1, train['y'])
test_label = np.where(test['y']==1, -1, test['y'])   ## -1 is known anomaly, 0 is unknown, 1 is known normal

from deepod.models.dsad import DeepSAD

clf = DeepSAD(epochs=100, rep_dim=128, batch_size=64, device='cpu', random_state=0)
clf.fit(train_data, train_label)

true = test['y']
pred = clf.predict(test_data)
score = clf.decision_function(test_data)

acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
result_all.append(['multivariate', 'DeepSAD', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])



# # unsupervised
# ## data
# from sklearn.decomposition import PCA

# train_data, test_data = [], []
# for i in tqdm(range(train_df.shape[0]-24)):
#     if 1 in train_label.iloc[i:i+24]['label'].values:
#         continue
#     else:
#         train_data.append(train_df.iloc[i:i+24].values.reshape(-1)) ## make train data into vector format with reshape(-1)
# for i in tqdm(range(test_df.shape[0]-24)):
#     test_data.append(test_df.iloc[i:i+24].values.reshape(-1))
# train_data = np.stack(train_data)
# test_data = np.stack(test_data)

# pca_1d = PCA(n_components=32)
# train_data_1d = pca_1d.fit_transform(train_data)
# test_data_1d = pca_1d.transform(test_data)


# ### OCSVM
# from sklearn.svm import OneClassSVM

# clf = OneClassSVM(kernel='rbf').fit(train_data_1d)

# true = test_label.iloc[24:].label.values
# pred = clf.predict(test_data_1d)
# pred = [1 if p==-1 else 0 for p in pred]

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['multivariate', 'OCSVM', rec, far, pre, rec, acc, auc, macro_f1, ap])


# ### LOF
# from sklearn.neighbors import LocalOutlierFactor

# clf = LocalOutlierFactor(n_neighbors=2, novelty=True)
# clf.fit(train_data_1d)

# true = test_label.iloc[24:].label.values
# pred = clf.predict(test_data_1d)
# pred = [1 if p==-1 else 0 for p in pred]

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['multivariate', 'LOF', rec, far, pre, rec, acc, auc, macro_f1, ap])


# ### IF
# from sklearn.ensemble import IsolationForest

# clf = IsolationForest(random_state=0)
# clf.fit(train_data_1d)

# true = test_label.iloc[24:].label.values
# pred = clf.predict(test_data_1d)
# pred = [1 if p==-1 else 0 for p in pred]

# acc, auc, far, pre, rec, macro_f1, ap = evaluate(true, pred, adjust=True, plot=False, print_=False)
# result_all.append(['multivariate', 'IForest', rec, far, pre, rec, acc, auc, macro_f1, ap])






#############################################################

result_all = pd.DataFrame(result_all, columns=['Method', 'model', 'DR', 'far', 'precision', 'recall', 'acc', 'AUC', 'F1_macro', 'F1_weight', 'AP'])
result_all.to_csv(f'result/{dataset}_baselines.csv')