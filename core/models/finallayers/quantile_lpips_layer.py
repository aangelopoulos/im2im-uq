import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torch.nn as nn
from core.models.losses.pinball import PinballLoss
import pdb
import lpips

class QuantileRegressionLPIPSLayer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(QuantileRegressionLPIPSLayer, self).__init__()
        self.q_lo = params["q_lo"] 
        self.q_hi = params["q_hi"]
        self.params = params
        self.lpips_loss = lpips.LPIPS(net='alex', device=params['device'])
        self.lower = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.upper = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        output = torch.cat((self.lower(x).unsqueeze(1), self.prediction(x).unsqueeze(1), self.upper(x).unsqueeze(1)), dim=1)
        return output

    def quantile_regression_lpips_loss_fn(self, pred, target, params):
      q_lo_loss = PinballLoss(quantile=params["q_lo"])
      q_hi_loss = PinballLoss(quantile=params["q_hi"])

      loss = params['q_lo_weight'] * q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
         params['q_hi_weight'] * q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze()) + \
         params['mse_weight'] * (self.lpips_loss(pred[:,1,:,:,:].repeat(1,3,1,1), target.repeat(1,3,1,1), normalize=True).squeeze().mean())

      return loss

def quantile_regression_lpips_nested_sets_from_output(model, output, lam=None):
  if lam == None:
      if model.lhat == None:
          raise Exception("You have to specify lambda unless your model is already calibrated.")
      lam = model.lhat 
  output[:,0,:,:,:] = torch.minimum(output[:,0,:,:,:], output[:,1,:,:,:]-1e-6)
  output[:,2,:,:,:] = torch.maximum(output[:,2,:,:,:], output[:,1,:,:,:]+1e-6)
  upper_edge = lam * (output[:,2,:,:,:] - output[:,1,:,:,:]) + output[:,1,:,:,:] 
  lower_edge = output[:,1,:,:,:] - lam * (output[:,1,:,:,:] - output[:,0,:,:,:])

  return lower_edge, output[:,1,:,:,:], upper_edge 