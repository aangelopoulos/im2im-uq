import torch

#INN loss class
class INNLoss():

  def __init__(self, beta=0.10, reduction='mean'):
      self.beta = beta 
      assert 0 <= self.beta
      self.reduction = reduction

  def __call__(self, lower, upper, target):
      assert target.shape == lower.shape
      assert target.shape == upper.shape
      loss = torch.relu(target-upper).square() + torch.relu(lower-target).square() + self.beta * torch.abs(upper - lower) 

      if self.reduction == 'sum':
        loss = loss.sum()
      if self.reduction == 'mean':
        loss = loss.mean()

      return loss
