import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torch.nn as nn
from core.models.losses.inn import INNLoss
import pdb

class INNLayer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(INNLayer, self).__init__()
        self.beta = params["beta"] 
        self.params = params

        self.lower = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.upper = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        output = torch.cat((self.lower(x).unsqueeze(1), self.prediction(x).unsqueeze(1), self.upper(x).unsqueeze(1)), dim=1)
        return output

def inn_loss_fn(pred, target, params):
  inn_loss = INNLoss(beta=params["beta"])
  mse_loss = nn.MSELoss()

  loss = mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze()) + inn_loss( pred[:,0,:,:,:].squeeze(), pred[:,2,:,:,:].squeeze(), target.squeeze() )

  return loss

def inn_nested_sets_from_output(model, output, lam=None):
  if lam == None:
      if model.lhat == None:
          raise Exception("You have to specify lambda unless your model is already calibrated.")
      lam = model.lhat 
  output[:,0,:,:,:] = torch.minimum(output[:,0,:,:,:], output[:,1,:,:,:]-1e-6)
  output[:,2,:,:,:] = torch.maximum(output[:,2,:,:,:], output[:,1,:,:,:]+1e-6)
  upper_edge = lam * (output[:,2,:,:,:] - output[:,1,:,:,:]) + output[:,1,:,:,:] 
  lower_edge = output[:,1,:,:,:] - lam * (output[:,1,:,:,:] - output[:,0,:,:,:])

  return lower_edge, output[:,1,:,:,:], upper_edge 
