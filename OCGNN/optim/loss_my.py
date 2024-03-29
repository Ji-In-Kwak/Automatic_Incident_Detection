import torch    
import numpy as np
import torch.nn.functional as F
    
def loss_function(nu, data_center, outputs, radius=0, mask=None):
    dist,scores=anomaly_score(data_center,outputs,radius,mask)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def anomaly_score(data_center,outputs,radius=0,mask= None):
    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    # c=data_center.repeat(outputs[mask].size()[0],1)
    # res=outputs[mask]-c
    # res=torch.mean(res, 1, keepdim=True)
    # dist=torch.diag(torch.mm(res,torch.transpose(res, 0, 1)))

    scores = dist - radius ** 2
    return dist,scores

def SAD_loss_function(nu, data_center, outputs, labels, radius=0, mask=None):
    # optimization parameter
    eps = 1e-6

    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    scores = dist - radius ** 2
    scores_pos = torch.max(torch.zeros_like(scores), scores)
    losses = torch.where(labels == 0, scores_pos, (scores_pos + eps) ** (-labels.float()))
    loss = radius ** 2 + (1 / nu) * torch.mean(losses)

    return loss,dist,scores


def init_center(args, loader, model, mask = True, label=0, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    device = f'cuda:{args.gpu}'
    c = torch.zeros(args.n_hidden, device=device)
    
    model.eval()
    with torch.no_grad():
        outputs = []
        for data in loader:
            if mask:
                mask_ix = np.where(data.y == label)
            else:
                mask_ix = np.arange(len(data.y))
            outputs.append(model(data.to(device))[mask_ix])
        outputs = torch.cat(outputs)        
        # get the inputs of the batch

        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, acc,loss, model,data_center, radius,epoch,path):
        score = acc
        cur_loss=loss
        if (self.best_score is None) or (self.lowest_loss is None):
        #if self.lowest_loss is None:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.save_checkpoint(acc,loss,model,data_center, radius, epoch,path)
        #elif cur_loss > self.lowest_loss:
        elif (score < self.best_score) and (cur_loss > self.lowest_loss):
            self.counter += 1
            if self.counter >= 0.8*(self.patience):
                print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.save_checkpoint(acc,loss,model,data_center, radius, epoch,path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, acc,loss,model,data_center, radius, epoch, path):
        '''Saves model when validation loss decrease.'''
        print('model saved. loss={:.4f} AUC={:.4f}'. format(loss,acc))
        # torch.save(model.state_dict(), path)
        torch.save({'model': model.state_dict(), 'data_center':data_center, 'radius':radius, 'epoch':epoch}, path)