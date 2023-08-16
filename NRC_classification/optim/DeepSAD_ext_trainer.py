import time
import numpy as np
import torch
import logging
from sklearn.metrics import *
#from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F

from optim.loss_my import SAD_loss_function,init_center,get_radius,EarlyStopping

# from utils.evaluate import graph_anomaly_evaluate

def train(args, logger, train_loader, val_loader, test_loader, model, path):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    checkpoints_path=path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')
    logger.info(f'dataset:{args.dataset}, normalize:{args.normalize}, exp_name:{args.exp_name}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    
    best_val_loss = 99999
    best_val_auc = 0

    # initialize data center
    data_center_0 = init_center(args, train_loader[0], model, mask=True, label=0)
    data_center_1 = init_center(args, train_loader[1], model, mask=True, label=0)
    data_center = torch.mean(torch.stack([data_center_0, data_center_1]), dim=0)
    # data_center= torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.

    
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)

    dur = []
    logger.info('Starting training...')
    model.to(device)
    model.train()
    for epoch in range(args.n_epochs):
        t0 = time.time()
        loss_sum = 0.0
        dist_all = []
        # forward
        for i in range(len(train_loader)):
            for (batch_idx, batch_graph) in enumerate(train_loader[i]):
                batch_graph = batch_graph.to(device)
                outputs = model(batch_graph)
                labels = batch_graph.y

                loss, dist, score = SAD_loss_function(args.nu, data_center, outputs, labels, radius)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                dist_all.append(dist)


        # epoch train loss
        train_loader_sum = np.sum([len(train_loader[i]) for i in range(len(train_loader))])
        train_loss = loss_sum / train_loader_sum
        arr_loss[epoch] = train_loss
        dur.append(time.time() - t0)
        dist_all = torch.concat(dist_all)

        # update the radius
        if epoch % 5 == 0:
#             radius.data=torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')
            radius.data=torch.tensor(get_radius(dist_all, args.nu), device=f'cuda:{args.gpu}')
            print("new radius = ", radius)
        
        # if ((epoch+1) % 50 == 0) or (epoch == 0):
        #     epoch_ckpt = checkpoints_path.replace('+bestcheckpoint.pt', f'+epoch{epoch+1}.pt')
        #     torch.save({'model': model.state_dict(), 'data_center':data_center, 'radius':radius, 'epoch':epoch}, epoch_ckpt)

        
        auc,ap,f1,acc,precision,recall,val_loss = graph_anomaly_evaluate(args,checkpoints_path, model, data_center, val_loader, radius)
        arr_valauc[epoch] = auc
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | ". format(epoch, np.mean(dur), 
                                                                                                                  train_loss,
                                                                                                                  val_loss, auc))
        logger.info("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | ". format(epoch, np.mean(dur), 
                                                                                                                  train_loss,
                                                                                                                  val_loss, auc))
        # print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | ". format(epoch, np.mean(dur), loss.item()*100000))
        
        # save model
        if auc == 0.0:
            if val_loss <= best_val_loss:
                print("Saving Model Parameters")
                torch.save({'model': model.state_dict(), 'data_center':data_center, 'radius':radius, 'epoch':epoch}, checkpoints_path)
                best_val_loss = val_loss
        else:
            if auc > best_val_auc:
                print("Saving Model Parameters")
                logger.info("Saving Model Parameters")
                torch.save({'model': model.state_dict(), 'data_center':data_center, 'radius':radius, 'epoch':epoch}, checkpoints_path)
                best_val_auc = auc
                                            
        
        if args.early_stop:
            if stopper.step(auc,val_loss.item(), model,data_center, radius,epoch,checkpoints_path):   
                break

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path)['model']) 

    
    auc,ap,f1,acc,precision,recall,loss = graph_anomaly_evaluate(args,checkpoints_path,model, data_center, test_loader, radius, mode='test')
    test_dur = 0
    arr_testauc[epoch]=auc
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur,auc,ap))
    # print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    logger.info('\n')

    #np.savez('SAGE-2.npz',epoch=arr_epoch,loss=arr_loss,valauc=arr_valauc,testauc=arr_testauc)

    return model


def graph_anomaly_evaluate(args,path, model, data_center,dataloader,radius,mode='val'):
    '''
    evaluate function
    '''
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if mode=='test':
        print(f'model loaded.')
        model.load_state_dict(torch.load(path)['model'])
    model.eval()
    total_loss=0
    # pred_list=[]
    # labels_list=[]
    # scores_list=[]
    #correct_label = 0
    with torch.no_grad():
        for i in range(len(dataloader)):
            for batch_idx, (batch_graph) in enumerate(dataloader[i]):
                batch_graph = batch_graph.to(device)
                outputs= model(batch_graph)

                # normlizing = nn.BatchNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
                # input_attr=normlizing(batch_graph.ndata['node_attr'])
                # outputs = model(batch_graph,input_attr)

                labels = batch_graph.y           
                loss, _, scores = SAD_loss_function(args.nu,data_center,outputs,labels,radius,None)

                labels = labels.cpu().numpy().astype('int8')
                #dist=dist.cpu().numpy()
                scores = scores.cpu().numpy()
                pred = thresholding(scores,0)

                total_loss+=loss.item()
                if batch_idx==0:
                    labels_vec=labels
                    pred_vec=pred
                    scores_vec=scores
                else:
                    pred_vec=np.append(pred_vec,pred)
                    labels_vec=np.concatenate((labels_vec,labels),axis=0)
                    scores_vec=np.concatenate((scores_vec,scores),axis=0)
                
                
        total_loss /= (len(dataloader[0])+len(dataloader[1]))
        # print('score std',scores_vec.std())
        # print('score mean',scores_vec.mean())
        # print('labels mean',labels_vec.mean())
        # print('pred mean',pred_vec.mean())
        if 1 not in labels_vec:
            acc=accuracy_score(labels_vec,pred_vec)
            recall=recall_score(labels_vec,pred_vec)
            precision=precision_score(labels_vec,pred_vec)
            f1=f1_score(labels_vec,pred_vec)
            return 0.0,0.0,f1,acc,precision,recall,total_loss

        else:
            auc=roc_auc_score(labels_vec, scores_vec)
            ap=average_precision_score(labels_vec, scores_vec)

            acc=accuracy_score(labels_vec,pred_vec)
            recall=recall_score(labels_vec,pred_vec)
            precision=precision_score(labels_vec,pred_vec)
            f1=f1_score(labels_vec,pred_vec)

            return auc,ap,f1,acc,precision,recall,total_loss

def thresholding(recon_error,threshold):
    ano_pred=np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i]>threshold:
            ano_pred[i]=1
    return ano_pred