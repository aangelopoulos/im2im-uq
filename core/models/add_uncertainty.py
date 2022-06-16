import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torch.nn as nn
from core.models.finallayers.quantile_layer import QuantileRegressionLayer, quantile_regression_loss_fn, quantile_regression_nested_sets_from_output
from core.models.finallayers.quantile_l1_layer import QuantileRegressionL1Layer, quantile_regression_l1_loss_fn, quantile_regression_l1_nested_sets_from_output
from core.models.finallayers.gaussian_layer import GaussianRegressionLayer, gaussian_regression_loss_fn, gaussian_regression_nested_sets_from_output
from core.models.finallayers.residual_magnitude_layer import ResidualMagnitudeLayer, residual_magnitude_loss_fn, residual_magnitude_nested_sets_from_output
from core.models.finallayers.residual_magnitude_l1_layer import ResidualMagnitudeL1Layer, residual_magnitude_l1_loss_fn, residual_magnitude_l1_nested_sets_from_output
from core.models.finallayers.inn_layer import INNLayer, inn_loss_fn, inn_nested_sets_from_output
from core.models.finallayers.softmax_layer import SoftmaxLayer, softmax_loss_fn, softmax_nested_sets_from_output
from core.utils import standard_to_minmax
import json

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

  # Always outputs [0,1] valued nested sets
  def nested_sets_from_output(self, output, lam=None):
    lower_edge, prediction, upper_edge = self.in_nested_sets_from_output_fn(self, output, lam)
    upper_edge = torch.maximum(upper_edge, prediction + 1e-6) # set a lower bound on the size.
    lower_edge = torch.minimum(lower_edge, prediction - 1e-6)

    return lower_edge, prediction, upper_edge 

  def nested_sets(self, x, lam=None):
    if lam == None:
        if self.lhat == None:
            raise Exception("You have to specify lambda unless your model is already calibrated.")
        lam = self.lhat 
    output = self(*x)
    return self.nested_sets_from_output(output, lam=lam)

  def set_lhat(self,lhat):
    self.lhat = lhat

def add_uncertainty(model, params): 
  last_layer = None
  train_loss_fn = None
  nested_sets_from_output_fn = None
  
  if params["uncertainty_type"] == "quantiles":
    last_layer = QuantileRegressionLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = quantile_regression_loss_fn    
    nested_sets_from_output_fn = quantile_regression_nested_sets_from_output
  elif params["uncertainty_type"] == "quantiles_l1":
    last_layer = QuantileRegressionL1Layer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = quantile_regression_l1_loss_fn    
    nested_sets_from_output_fn = quantile_regression_l1_nested_sets_from_output
  elif params["uncertainty_type"] == "gaussian":
    last_layer = GaussianRegressionLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = gaussian_regression_loss_fn    
    nested_sets_from_output_fn = gaussian_regression_nested_sets_from_output
  elif params["uncertainty_type"] == "residual_magnitude":
    last_layer = ResidualMagnitudeLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = residual_magnitude_loss_fn    
    nested_sets_from_output_fn = residual_magnitude_nested_sets_from_output
  elif params["uncertainty_type"] == "residual_magnitude_l1":
    last_layer = ResidualMagnitudeL1Layer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = residual_magnitude_l1_loss_fn    
    nested_sets_from_output_fn = residual_magnitude_l1_nested_sets_from_output
  elif params["uncertainty_type"] == "softmax":
    last_layer = SoftmaxLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = softmax_loss_fn
    nested_sets_from_output_fn = softmax_nested_sets_from_output
  elif params["uncertainty_type"] == "inn":
    last_layer = INNLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = inn_loss_fn
    nested_sets_from_output_fn = inn_nested_sets_from_output
  else:
    raise NotImplementedError

  return ModelWithUncertainty(model, last_layer, train_loss_fn, nested_sets_from_output_fn, params)
