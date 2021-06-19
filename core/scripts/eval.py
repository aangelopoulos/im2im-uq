import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from core.calibration.calibrate_model import get_rcps_loss_fn, get_rcps_losses_and_sizes_from_outputs
import pdb

def eval_risk_size(model, dataset, config):
  with torch.no_grad():
    model.eval()
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    outputs = torch.cat([model(x[0].unsqueeze(0).to(device)) for x in dataset], dim=0)
    labels = torch.cat([x[1].unsqueeze(0).to(device) for x in dataset], dim=0)
    out_dataset = TensorDataset(outputs,labels)
    losses, sizes = get_rcps_losses_and_sizes_from_outputs(model, out_dataset, rcps_loss_fn, device)
    return losses.mean(), sizes

def eval_net(net, loader, device):
    with torch.no_grad():
      net.eval()
      net.to(device=device)
      #label_type = torch.float32 if net.n_classes == 1 else torch.long
      n_val = len(loader)  # the number of batch
      if n_val == 0:
          print("No val points, returning 0")
          return 0

      val_loss = 0

      with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
          for batch in loader:
              labels = batch[-1].to(device=device)
              x = tuple([batch[i] for i in range(len(batch)-1)])
              x = [x[i].to(device=device, dtype=torch.float32) for i in range(len(x))]

              # Predict
              labels_pred = net(*x) # Unpack tuple

              val_loss += net.loss_fn(labels_pred, labels).item()/n_val
              pbar.update()

      net.train()
      return val_loss 
