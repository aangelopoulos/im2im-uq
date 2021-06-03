import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torch.nn as nn
from core.models.losses.pinball import PinballLoss

class QuantileRegressionLayer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(QuantileRegressionLayer, self).__init__()
        self.q_lo = params["q_lo"] 
        self.q_hi = params["q_hi"]
        self.params = params

        self.lower = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.upper = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        output = torch.cat((self.lower(x), self.prediction(x), self.upper(x)), dim=1)
        return output

def quantile_regression_loss_fn(pred, target, params):
  q_lo_loss = PinballLoss(quantile=params["q_lo"])
  q_hi_loss = PinballLoss(quantile=params["q_hi"])
  mse_loss = nn.MSELoss()

  loss = params['q_lo_weight'] * q_lo_loss(pred[:,0,:,:], target.squeeze()) + \
         params['q_hi_weight'] * q_hi_loss(pred[:,2,:,:], target.squeeze()) + \
         params['mse_weight'] * mse_loss(pred[:,1,:,:], target.squeeze())

  return loss

def quantile_regression_nested_sets(model, x, lam=None):
  if lam == None:
      if model.lhat == None:
          raise Exception("You have to specify lambda unless your model is already calibrated.")
      lam = model.lhat 
  output = model(x)
  return model.nested_sets_from_output(output,lam=lam)

def quantile_regression_nested_sets_from_output(model, output, lam=None):
  if lam == None:
      if model.lhat == None:
          raise Exception("You have to specify lambda unless your model is already calibrated.")
      lam = model.lhat 
  upper_edge = lam * (output[:,2,:,:] - output[:,1,:,:]) + output[:,1,:,:] 
  lower_edge = output[:,1,:,:] - lam * (output[:,1,:,:] - output[:,0,:,:])
  upper_edge = torch.maximum(upper_edge, output[:,1,:,:] + 1e-1) # set a lower bound on the size.
  lower_edge = torch.minimum(lower_edge, output[:,1,:,:] - 1e-1)
  return lower_edge, upper_edge 
