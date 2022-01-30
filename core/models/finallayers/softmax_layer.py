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
  target[target >= params["num_softmax"]] = params["num_softmax"]-1

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

    # Calculate the prediction intervals
    cumsum = torch.cumsum(output,dim=1)

    lower_quantile = (cumsum <= 0.05).float().sum(dim=1)/num_softmax
    upper_quantile = (cumsum <= 0.95).float().sum(dim=1)/num_softmax
    prediction = torch.argmax(output, dim=1)/num_softmax

    # Ensure the sets dont collapse
    lower_quantile[prediction==lower_quantile] -= 1/num_softmax
    upper_quantile[prediction==upper_quantile] += 1/num_softmax
    lower_quantile = lower_quantile.clamp(min=0,max=1)
    upper_quantile = upper_quantile.clamp(min=0,max=1)

    lower_edge = prediction - (prediction-lower_quantile).relu()*lam
    upper_edge = prediction + (upper_quantile-prediction).relu()*lam

    return lower_edge, prediction, upper_edge 
