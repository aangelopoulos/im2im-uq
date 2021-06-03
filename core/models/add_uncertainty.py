import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
from core.models.finallayers.quantile_layer import QuantileRegressionLayer
from core.models.trunks.wnet import WNet


def add_uncertainty(model, params): 
  base_model_type = None
  last_layer = None
  train_loss_fn = None
  nested_sets_fn = None
  nested_sets_from_output_fn = None

  # Get the trunk
  if params["model_type"] == "WNet":
    base_model_type = WNet
  else:
    raise NotImplementedError
  
  if params["uncertainty_type"] == "quantiles":
    last_layer = QuantileRegressionLayer(model.n_channels_middle, model.n_channels_out, params) 
    train_loss_fn = quantile_regression_loss_fn    
    nested_sets_fn = quantile_regression_nested_sets
    nested_sets_from_output_fn = quantile_regression_nested_sets_from_output
  else:
    raise NotImplementedError

  class _model_with_uncertainty(WNet):
    def __init__(self, baseModel):
        self.__class__ = type(baseModel.__class__.__name__,
                              (self.__class__, baseModel.__class__),
                              {})
        self.__dict__ = baseModel.__dict__
        self.trunk = baseModel
  
    def forward(self, x):
      x = self.trunk(x)
      return last_layer(x)

    def loss_fn(self, pred, target, params):
      return train_loss_fn(self,pred,target,params) 

    def nested_sets(self, x, lam=None):
      return nested_sets_fn(self, x, lam)

    def nested_sets_from_output_fn(self, output, lam=None):
      return nested_sets_from_output_fn(self, output, lam)

  return _model_with_uncertainty(model)
