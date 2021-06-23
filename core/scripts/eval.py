import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from core.calibration.calibrate_model import get_rcps_loss_fn, get_rcps_metrics_from_outputs
import wandb
import pdb

def transform_output(x):
  x = np.maximum(0,np.minimum(255*x.cpu().squeeze(), 255))
  if len(x.shape) == 3:
    x = x.permute(1,2,0)
  return x.numpy().astype(np.uint8)

def get_images(model,
               val_dataset,
               device,
               idx_iterator):
  with torch.no_grad():
    lam = None
    if model.lhat == None:
      lam = 1.0
    examples_input = [wandb.Image(transform_output(val_dataset[img_idx][0])) for img_idx in idx_iterator]
    examples_ground_truth = [wandb.Image(transform_output(val_dataset[img_idx][1])) for img_idx in idx_iterator]
    examples_output = [model.nested_sets((val_dataset[img_idx][0].unsqueeze(0).to(device),),lam=lam) for img_idx in idx_iterator]
    examples_lower_edge = [wandb.Image(transform_output(example[0])) for example in examples_output]
    examples_prediction = [wandb.Image(transform_output(example[1])) for example in examples_output]
    examples_upper_edge = [wandb.Image(transform_output(example[2])) for example in examples_output]
    return examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth

def eval_set_metrics(model, dataset, config):
  with torch.no_grad():
    model.eval()
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    labels = torch.cat([x[1].unsqueeze(0).to(device) for x in dataset], dim=0)
    outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
    outputs_shape[0] = len(dataset)
    outputs = torch.zeros(tuple(outputs_shape),device=device)
    for i in range(len(dataset)):
      outputs[i,:,:,:,:] = model(dataset[i][0].unsqueeze(0).to(device))
    out_dataset = TensorDataset(outputs,labels)
    losses, sizes, spearman, stratified_risks = get_rcps_metrics_from_outputs(model, out_dataset, rcps_loss_fn, device)
    return losses.mean(), sizes, spearman, stratified_risks

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
