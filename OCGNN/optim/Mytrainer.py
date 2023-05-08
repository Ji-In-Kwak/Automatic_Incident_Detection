import time
import numpy as np
import torch
import logging
#from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F

from optim.loss_my import loss_function,init_center,get_radius,EarlyStopping

from utils.evaluate import node_anomaly_evaluate, graph_anomaly_evaluate

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
    best_val_loss = 99
    best_val_auc = 0
    # initialize data center

    data_center= init_center(args, train_loader, model)
    # data_center= torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.

    
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)

    dur = []
    model.to(device)
    model.train()
    for epoch in range(args.n_epochs):
        t0 = time.time()
        loss_sum = 0.0
        # forward
        for (batch_idx, batch_graph) in enumerate(train_loader):
            batch_graph = batch_graph.to(device)
            outputs= model(batch_graph)
        #print('model:',args.module)
        #print('output size:',outputs.size())
        
            loss, dist, score=loss_function(args.nu, data_center, outputs, radius)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        # epoch train loss
        arr_loss[epoch] = loss_sum / len(train_loader)
        dur.append(time.time() - t0)
        if epoch % 3 == 0:
            radius.data=torch.tensor(get_radius(dist, args.nu), device=f'cuda:{args.gpu}')
            print("new radius = ", radius)
        # if ((epoch+1) % 50 == 0) or (epoch == 0):
        #     epoch_ckpt = checkpoints_path.replace('+bestcheckpoint.pt', f'+epoch{epoch+1}.pt')
        #     torch.save({'model': model.state_dict(), 'data_center':data_center, 'radius':radius, 'epoch':epoch}, epoch_ckpt)

        
        auc,ap,f1,acc,precision,recall,val_loss = graph_anomaly_evaluate(args,checkpoints_path, model, data_center, val_loader, radius)
        arr_valauc[epoch] = auc
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | ". format(epoch, np.mean(dur), 
                                                                                                                  loss.item()*100000,
                                                                                                                  val_loss.item()*100000, auc))
        # print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | ". format(epoch, np.mean(dur), loss.item()*100000))
        
        # save model
        # if val_loss <= best_val_loss:
        #     print("Saving Model Parameters")
        #     torch.save({'model': model.state_dict(), 'data_center':data_center, 'radius':radius, 'epoch':epoch}, checkpoints_path)
        #     best_val_loss = val_loss
        if auc > best_val_auc:
            print("Saving Model Parameters")
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


