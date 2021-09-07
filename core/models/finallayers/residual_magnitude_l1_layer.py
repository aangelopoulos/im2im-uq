import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import torch
import torch.nn as nn
import pdb

class ResidualMagnitudeL1Layer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, params):
        super(ResidualMagnitudeL1Layer, self).__init__()
        self.params = params

        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.residual_magnitude = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(self, x):
        output = torch.cat((self.prediction(x).unsqueeze(1), (self.residual_magnitude(x).unsqueeze(1)).abs()), dim=1)
        return output

def residual_magnitude_l1_loss_fn(pred, target, params):
  criterion1 = nn.L1Loss()
  criterion2 = nn.MSELoss()

  loss = criterion1(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
         criterion2(pred[:,1,:,:,:].squeeze(), (target.squeeze() - pred[:,0,:,:,:].squeeze()).abs())

  return loss

def residual_magnitude_l1_nested_sets_from_output(model, output, lam=None):
  if lam == None:
      if model.lhat == None:
          raise Exception("You have to specify lambda unless your model is already calibrated.")
      lam = model.lhat 
  upper_edge = lam * output[:,1,:,:,:] + output[:,0,:,:,:]
  lower_edge = -lam * output[:,1,:,:,:] + output[:,0,:,:,:]

  return lower_edge, output[:,0,:,:,:], upper_edge 
