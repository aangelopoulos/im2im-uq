import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pdb

def get_rcps_losses(model, dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0) 
  for batch in dataloader:
    sets = model.nested_sets_from_output(batch,lam) 
    losses = losses + [rcps_loss_fn(sets, labels),]
  return torch.cat(losses,dim=0)

def get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(out_dataset, batch_size=32, shuffle=False, num_workers=0) 
  for batch in dataloader:
    x, labels = batch
    sets = model.nested_sets_from_output(x,lam) 
    losses = losses + [rcps_loss_fn(sets, labels),]
  return torch.cat(losses,dim=0)

def fraction_missed_loss(pset,label):
  misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[1].squeeze() < label.squeeze()).float()
  d = len(misses.shape)
  return misses.mean(dim=tuple(range(1,d)))

def get_rcps_loss_fn(config):
  string = config['rcps_loss']
  if string == 'fraction_missed':
    return fraction_missed_loss
  else:
    raise NotImplementedError

def calibrate_model(model, dataset, config):
  with torch.no_grad():
    alpha = config['alpha']
    delta = config['delta']
    device = config['device']
    lambdas = torch.linspace(0,config['maximum_lambda'],config['num_lambdas'])
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).cpu() for x in dataset], dim=0)
    labels = torch.cat([x[1].unsqueeze(0).cpu() for x in dataset], dim=0)
    out_dataset = TensorDataset(outputs,labels)
    print("Calibrating...")
    for lam in reversed(lambdas):
      losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam-1/config['num_lambdas'], device)
      Rhat = losses.mean()
      print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}",end='')
      if Rhat > alpha: # TODO: Replace with concentration
        model.lhat = lam
        return 
