import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torch.nn as nn
import pdb

class SoftmaxLayer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(SoftmaxLayer, self).__init__()
        self.num_softmax = params["num_softmax"] 
        self.output_layers = nn.ModuleList([nn.Conv2d(n_channels_middle, self.num_softmax, kernel_size=3, padding=1) for i in range(n_channels_out)])

    def forward(self, x):
        return torch.cat([ layer(x).unsqueeze(2) for layer in self.output_layers ], dim=1)

def softmax_loss_fn(pred, target, params):
  criterion = nn.CrossEntropyLoss()

  classes = torch.linspace(0,1,params["num_softmax"], device=params["device"])
  target = torch.bucketize(target,classes,right=False) 

  loss = criterion(pred,target)

  return loss

def softmax_nested_sets_from_output(model, output, lam=None):
  with torch.no_grad():
    if lam == None:
        if model.lhat == None:
            raise Exception("You have to specify lambda unless your model is already calibrated.")
        lam = model.lhat
    
    output = output.softmax(dim=1)
    num_softmax = output.shape[1]

    # The Romano et al. version of classification
    vals, idxs = output.sort(dim=1,descending=True)
    torch.cumsum(vals,dim=1,out=vals)
    idxs[vals > lam] = -num_softmax
    lower_edge = idxs.abs().min(dim=1)[0]/num_softmax
    prediction = torch.argmax(output, dim=1)/num_softmax
    upper_edge = idxs.max(dim=1)[0]/num_softmax

    upper_edge = torch.maximum(upper_edge, prediction + 1e-6) # set a lower bound on the size.
    lower_edge = torch.minimum(lower_edge, prediction - 1e-6)
    upper_edge[upper_edge > 1] = 1.0
    lower_edge[lower_edge < 0] = 0.0
    return lower_edge, prediction, upper_edge 
