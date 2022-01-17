import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torch.nn as nn
import pdb

class GaussianRegressionLayer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(GaussianRegressionLayer, self).__init__()
        self.params = params

        self.mean = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.variance = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        output = torch.cat((self.mean(x).unsqueeze(1), torch.relu(self.variance(x).unsqueeze(1))), dim=1)
        return output

def gaussian_regression_loss_fn(pred, target, params):
  criterion = nn.GaussianNLLLoss()

  loss = criterion(pred[:,0,:,:,:].squeeze(), target.squeeze(), pred[:,1,:,:,:].squeeze())

  return loss

def gaussian_regression_nested_sets_from_output(model, output, lam=None):
  if lam == None:
      if model.lhat == None:
          raise Exception("You have to specify lambda unless your model is already calibrated.")
      lam = model.lhat 
  upper_edge = lam * output[:,1,:,:,:].sqrt() + output[:,0,:,:,:]
  lower_edge = -lam * output[:,1,:,:,:].sqrt() + output[:,0,:,:,:]

  return lower_edge, output[:,0,:,:,:], upper_edge 
