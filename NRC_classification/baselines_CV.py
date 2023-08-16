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
import argparse


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


parser = argparse.ArgumentParser(description='desc')
parser.add_argument('--target-sid', required=True, type=int, help='incident road sid')
parser.add_argument('--data-type', required=True, type=str, help='data preprocessing type')
parser.add_argument('--num-cv', required=True, type=int, help='total number of cross validation set')
args = parser.parse_args()


# target_sid = 1210005301  ## 1210005301  ## 1030001902  ## 1220005401  ## 1210003000  ## 1130052300
# accident_case = accident_all[accident_all.loc[:, 'accident_sid'] == target_sid]
# eventID = accident_case.eventId.iloc[0]


target_sid = args.target_sid
data_type = args.data_type
dataset = '{}_{}'.format(target_sid, data_type)
print('dataset = ', dataset)


## Evaluation
def adjust_predictions(predictions, labels):
    adjustment_started = False
    new_pred = predictions

    for i in range(len(predictions)):
        if (labels[i] == 1) & (predictions[i] == 1):
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


result_all = []


## random Cross-validation

for k in range(args.num_cv):
    ## Data Loading
    print('Data Loading .... CV = ', k)

    train = np.load('../data/{}/train{}.npz'.format(dataset, k))
    val = np.load('../data/{}/val{}.npz'.format(dataset, k))
    test = np.load('../data/{}/test{}.npz'.format(dataset, k))

    H = nx.read_gpickle("../data/{}/sensor_graph.gpickle".format(dataset))


    ##################################
    ## Multivariate ###
    ##################################



    # DATA
    train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
    test_data = test['x'].reshape(test['x'].shape[0], -1)

    train_label = train['y']
    test_label = test['y']


    # SVM
    print('SVM')
    from sklearn.svm import SVC

    clf = SVC(probability=True)
    clf.fit(train_data, train_label)

    true = test_label
    pred = clf.predict(test_data)
    score = clf.predict_proba(test_data)[:, 1]

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    print('AUC score = ', auc)
    result_all.append([k, 'multivariate', 'SVM', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


    # MLP
    print('MLP')
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(64, 64))
    clf.fit(train_data, train_label)

    true = test_label
    pred = clf.predict(test_data)
    score = clf.predict_proba(test_data)[:, 1]

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    print('AUC score = ', auc)
    result_all.append([k, 'multivariate', 'MLP', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])





    ## multivariate 3D data
    train_data = np.transpose(train['x'], (0, 2, 1))  ## (n_samples, n_node, seq_len)
    test_data = np.transpose(test['x'], (0, 2, 1))

    train_label = train['y']
    test_label = test['y']


    # MLP
    print('Multivariate MLP')
    from sktime.classification.deep_learning import MLPClassifier

    clf = MLPClassifier(n_epochs=200, batch_size=64, random_state=0)
    clf.fit(train_data, train_label)

    true = test_label
    pred = clf.predict(test_data)
    score = clf.predict_proba(test_data)[:, 1]

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    result_all.append([k, 'multivariate', 'Time-MLP', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])



    # CNN
    print('Multivariate CNN')
    from sktime.classification.deep_learning import CNNClassifier

    clf = CNNClassifier(n_epochs=200, batch_size=64, random_state=0)
    clf.fit(train_data, train_label)

    true = test_label
    pred = clf.predict(test_data)
    score = clf.predict_proba(test_data)[:, 1]

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    result_all.append([k, 'multivariate', 'Time-CNN', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


    # LSTM-FCN
    print('Multivariate LSTM-FCN')
    from sktime.classification.deep_learning import LSTMFCNClassifier

    clf = LSTMFCNClassifier(n_epochs=200, batch_size=64, random_state=0, verbose=True)
    clf.fit(train_data, train_label)

    true = test_label
    pred = clf.predict(test_data)
    score = clf.predict_proba(test_data)[:, 1]

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    result_all.append([k, 'multivariate', 'LSTM-FCN', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])



    # DevNet
    print('DevNet')

    # DATA
    train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
    test_data = test['x'].reshape(test['x'].shape[0], -1)

    train_label = np.where(train['y']==1, 1, train['y'])
    test_label = np.where(test['y']==1, 1, test['y'])   ## 1 is known anomaly, 0 is unknown

    from deepod.models.devnet import DevNet

    clf = DevNet(epochs=200, batch_size=64, device='cpu', random_state=0)
    clf.fit(train_data, train_label)

    true = test['y']
    pred = clf.predict(test_data)
    score = clf.decision_function(test_data)

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    result_all.append([k, 'multivariate', 'DevNet', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])



    # DeepSAD
    print('Raw DeepSAD')

    # DATA
    train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
    test_data = test['x'].reshape(test['x'].shape[0], -1)

    train_label = np.where(train['y']==1, -1, train['y'])
    test_label = np.where(test['y']==1, -1, test['y'])   ## -1 is known anomaly, 0 is unknown, 1 is known normal

    from deepod.models.dsad import DeepSAD

    clf = DeepSAD(epochs=200, rep_dim=128, batch_size=64, device='cpu', random_state=0)
    clf.fit(train_data, train_label)

    true = test['y']
    pred = clf.predict(test_data)
    score = clf.decision_function(test_data)

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    result_all.append([k, 'multivariate', 'DeepSAD', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


#     # FeaWAD
#     print('FeaWAD')

#     # DATA
#     train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
#     test_data = test['x'].reshape(test['x'].shape[0], -1)

#     train_label = np.where(train['y']==1, 1, train['y'])
#     test_label = np.where(test['y']==1, 1, test['y'])   ## 1 is known anomaly, 0 is unknown

#     from deepod.models.feawad import FeaWAD

#     clf = FeaWAD(epochs=200, batch_size=64, device='cpu', random_state=0)
#     clf.fit(train_data, train_label)

#     true = test['y']
#     pred = clf.predict(test_data)
#     score = clf.decision_function(test_data)

#     acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
#     result_all.append([k, 'multivariate', 'FeaWAD', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])

    

    # PReNet
    print('PReNet')

    # DATA
    train_data = train['x'].reshape(train['x'].shape[0], -1)  ## (n_samples, seq_len * n_node)
    test_data = test['x'].reshape(test['x'].shape[0], -1)

    train_label = np.where(train['y']==1, 1, train['y'])
    test_label = np.where(test['y']==1, 1, test['y'])   ## 1 is known anomaly, 0 is unknown

    from deepod.models.prenet import PReNet

    clf = PReNet(epochs=50, batch_size=64, device='cpu', random_state=0)
    clf.fit(train_data, train_label)

    true = test['y']
    pred = clf.predict(test_data)
    score = clf.decision_function(test_data)

    acc, auc, far, pre, rec, macro_f1, weight_f1, ap = evaluate(true, pred, score, adjust=False, plot=False, print_=False)
    result_all.append([k, 'multivariate', 'PReNet', rec, far, pre, rec, acc, auc, macro_f1, weight_f1, ap])


#############################################################

os.makedirs('result/', exist_ok=True)
result_all = pd.DataFrame(result_all, columns=['Kfold', 'Method', 'model', 'DR', 'far', 'precision', 'recall', 'acc', 'AUC', 'F1_macro', 'F1_weight', 'AP'])
result_all.to_csv(f'result/{dataset}_baselines.csv')