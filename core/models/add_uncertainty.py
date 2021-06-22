import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torch.nn as nn
from core.models.finallayers.quantile_layer import QuantileRegressionLayer, quantile_regression_loss_fn, quantile_regression_nested_sets_from_output
from core.models.finallayers.gaussian_layer import GaussianRegressionLayer, gaussian_regression_loss_fn, gaussian_regression_nested_sets_from_output
from core.models.finallayers.residual_magnitude_layer import ResidualMagnitudeLayer, residual_magnitude_loss_fn, residual_magnitude_nested_sets_from_output
from core.models.finallayers.softmax_layer import SoftmaxLayer, softmax_loss_fn, softmax_nested_sets_from_output
from core.models.trunks.wnet import WNet
from core.models.trunks.unet import UNet
import jsons

class ModelWithUncertainty(nn.Module):
  def __init__(self, baseModel, last_layer, in_train_loss_fn, in_nested_sets_from_output_fn, params):
      super(ModelWithUncertainty, self).__init__()
      self.baseModel = baseModel
      self.last_layer = last_layer
      self.register_buffer('lhat',None)
      self.in_train_loss_fn = in_train_loss_fn
      self.in_nested_sets_from_output_fn = in_nested_sets_from_output_fn
      self.params = params
  
  def forward(self, x):
    x = self.baseModel(x)
    return self.last_layer(x)

  def loss_fn(self, pred, target):
    return self.in_train_loss_fn(pred,target,self.params) 

  def nested_sets_from_output(self, output, lam=None):
    return self.in_nested_sets_from_output_fn(self, output, lam)

  def nested_sets(self, x, lam=None):
    if lam == None:
        if self.lhat == None:
            raise Exception("You have to specify lambda unless your model is already calibrated.")
        lam = self.lhat 
    output = self(*x)
    return self.nested_sets_from_output(output, lam=lam)

def add_uncertainty(model, params): 
  base_model_type = None
  last_layer = None
  train_loss_fn = None
  nested_sets_from_output_fn = None

  # Get the trunk
  if params["model"] == "UNet":
    base_model_type = UNet
  elif params["model"] == "WNet":
    base_model_type = WNet
  else:
    raise NotImplementedError
  
  if params["uncertainty_type"] == "quantiles":
    last_layer = QuantileRegressionLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = quantile_regression_loss_fn    
    nested_sets_from_output_fn = quantile_regression_nested_sets_from_output
  elif params["uncertainty_type"] == "gaussian":
    last_layer = GaussianRegressionLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = gaussian_regression_loss_fn    
    nested_sets_from_output_fn = gaussian_regression_nested_sets_from_output
  elif params["uncertainty_type"] == "residual_magnitude":
    last_layer = ResidualMagnitudeLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = residual_magnitude_loss_fn    
    nested_sets_from_output_fn = residual_magnitude_nested_sets_from_output
  elif params["uncertainty_type"] == "softmax":
    last_layer = SoftmaxLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = softmax_loss_fn    
    nested_sets_from_output_fn = softmax_nested_sets_from_output
  else:
    raise NotImplementedError

  return ModelWithUncertainty(model, last_layer, train_loss_fn, nested_sets_from_output_fn, params)
