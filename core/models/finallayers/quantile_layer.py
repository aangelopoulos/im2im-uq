import torch
import torch.nn as nn

class Quantile_Regression_Layer(nn.Module):
    def __init__(self, n_channels_middle, n_channels_out, q_lo, q_hi, params):
        super(Quantile_Layer, self).__init__()
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.params = params

        self.lower = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.prediction = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)
        self.upper = nn.Conv2d(n_channels_middle, n_channels_out, kernel_size=3, padding=1)

    def forward(x):
        output = torch.cat((self.lower(x), self.prediction(x), self.upper(x)), dim=1)
        return output

def quantile_regression_loss_fn(pred, target, q_lo, q_hi, params):
  q_lo_loss = PinballLoss(quantile=q_lo)
  q_hi_loss = PinballLoss(quantile=q_hi)
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
