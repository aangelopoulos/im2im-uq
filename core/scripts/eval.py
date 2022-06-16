import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from core.calibration.calibrate_model import get_rcps_loss_fn, get_rcps_metrics_from_outputs
from core.utils import standard_to_minmax
import wandb
import pdb

def transform_output(x,self_normalize=True):
  if self_normalize:
    x = x - x.min()
    x = x / x.max()

  x = np.maximum(0,np.minimum(255*x.cpu().squeeze(), 255))
  if len(x.shape) == 3:
    x = x.permute(1,2,0)
  return x.numpy().astype(np.uint8)

def get_images(model,
               val_dataset,
               device,
               idx_iterator,
               config):
  with torch.no_grad():
    model = model.to(device)

    lam = None
    if model.lhat == None:
      if config["uncertainty_type"] != "softmax":
        lam = 1.0
      else:
        lam = 0.99

    try:
      # If dataset is iterable, create a list of outputs
      my_iter = iter(val_dataset)
      val_dataset = [next(my_iter) for img_idx in idx_iterator]
    except:
      pass

    examples_output = [model.nested_sets((val_dataset[img_idx][0].unsqueeze(0).to(device),),lam=lam) for img_idx in idx_iterator]
    examples_gt = [val_dataset[img_idx][1] for img_idx in idx_iterator]
    if val_dataset[0][0].shape[0] > 1:
      inputs = [val_dataset[img_idx][0][0] for img_idx in idx_iterator]
    else:
      inputs = [val_dataset[img_idx][0] for img_idx in idx_iterator]
    raw_images_dict = {'inputs': inputs, 
                       'gt': examples_gt,
                       'predictions': [example[1] for example in examples_output], 
                       'lower_edge': [example[0] for example in examples_output], 
                       'upper_edge': [example[2] for example in examples_output] 
                      }

    if val_dataset[0][0].shape[0] > 1:
      examples_input = [wandb.Image(transform_output(val_dataset[img_idx][0][0])) for img_idx in idx_iterator]
    else:
      examples_input = [wandb.Image(transform_output(val_dataset[img_idx][0])) for img_idx in idx_iterator]
    examples_lower_edge = [wandb.Image(transform_output(example[0])) for example in examples_output]
    examples_prediction = [wandb.Image(transform_output(example[1])) for example in examples_output]
    examples_upper_edge = [wandb.Image(transform_output(example[2])) for example in examples_output]
    examples_ground_truth = [wandb.Image(transform_output(val_dataset[img_idx][1])) for img_idx in idx_iterator]

    # Calculate lengths on their own scales
    lower_lengths = [example[1]-example[0] for example in examples_output]
    lower_lengths = [transform_output(lower_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()), self_normalize=False) for i in range(len(examples_output))]
    #lower_lengths = [lower_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()) for i in range(len(examples_output))]
    upper_lengths = [example[2]-example[1] for example in examples_output]
    upper_lengths = [transform_output(upper_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()), self_normalize=False) for i in range(len(examples_output))]
    #upper_lengths = [upper_lengths[i]/(examples_output[i][1].max()-examples_output[i][1].min()) for i in range(len(examples_output))]

    examples_lower_length = [wandb.Image(ll) for ll in lower_lengths]
    examples_upper_length = [wandb.Image(ul) for ul in upper_lengths]

    try:
      val_dataset.reset()
    except:
      pass

    return examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth, examples_lower_length, examples_upper_length, raw_images_dict

def get_loss_table(model, dataset, config):
  try:
    dataset.reset()
  except:
    print("dataset is map-style (not resettable)")
  with torch.no_grad():
    if config["uncertainty_type"] == "softmax":
      lambdas = torch.linspace(config['minimum_lambda_softmax'],config['maximum_lambda_softmax'],config['num_lambdas'])
    else:
      lambdas = torch.linspace(config['minimum_lambda'],config['maximum_lambda'],config['num_lambdas'])
    model.eval()
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    labels = torch.cat([x[1].unsqueeze(0).to(device).to('cpu') for x in dataset], dim=0).cpu()

    if config['dataset'] == 'temca':
      outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in dataset], dim=0)
    else:
      outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
      outputs_shape[0] = len(dataset)
      outputs = torch.zeros(tuple(outputs_shape),device='cpu')
      
      for i in range(len(dataset)):
        print(f"Validation output {i}")
        outputs[i,:,:,:,:] = model(dataset[i][0].unsqueeze(0).to(device)).cpu()
    out_dataset = TensorDataset(outputs,labels)

    print("GET LOSS TABLE FROM OUTPUTS")
    loss_table = torch.zeros((outputs.shape[0],config['num_lambdas']))
    dataloader = DataLoader(out_dataset, batch_size=4, shuffle=False, num_workers=0) 
    model = model.to(device)
    i = 0
    for batch in dataloader:
      x, labels = batch
      labels = labels.to(device)
      for j in range(lambdas.shape[0]):
        sets = model.nested_sets_from_output(x.to(device), lam=lambdas[j]) 
        loss_table[i:i+x.shape[0],j] = rcps_loss_fn(sets, labels)
      i += x.shape[0]
    print("DONE!")
    return loss_table 


def eval_set_metrics(model, dataset, config):
  try:
    dataset.reset()
  except:
    print("dataset is map-style (not resettable)")
  with torch.no_grad():
    model.eval()
    device = config['device']
    rcps_loss_fn = get_rcps_loss_fn(config)
    model = model.to(device)
    labels = torch.cat([x[1].unsqueeze(0).to(device).to('cpu') for x in dataset], dim=0).cpu()

    if config['dataset'] == 'temca':
      outputs = torch.cat([model(x[0].unsqueeze(0).to(device)).to('cpu') for x in dataset], dim=0)
    else:
      outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
      outputs_shape[0] = len(dataset)
      outputs = torch.zeros(tuple(outputs_shape),device='cpu')
      
      for i in range(len(dataset)):
        print(f"Validation output {i}")
        outputs[i,:,:,:,:] = model(dataset[i][0].unsqueeze(0).to(device)).cpu()
    out_dataset = TensorDataset(outputs,labels)

    print("GET RCPS METRICS FROM OUTPUTS")
    losses, sizes, spearman, stratified_risks, mse, spatial_miscoverage = get_rcps_metrics_from_outputs(model, out_dataset, rcps_loss_fn, device)
    print("DONE!")
    return losses.mean(), sizes, spearman, stratified_risks, mse, spatial_miscoverage

def eval_net(net, loader, device):
    with torch.no_grad():
      net.eval()
      net.to(device=device)
      #label_type = torch.float32 if net.n_classes == 1 else torch.long

      val_loss = 0
      num_val = 0

      with tqdm(total=10000, desc='Validation round', unit='batch', leave=False) as pbar:
          for batch in loader:
              labels = batch[-1].to(device=device)
              x = tuple([batch[i] for i in range(len(batch)-1)])
              x = [x[i].to(device=device, dtype=torch.float32) for i in range(len(x))]

              # Predict
              labels_pred = net(*x) # Unpack tuple

              num_val += labels.shape[0]
              val_loss += net.loss_fn(labels_pred, labels).item()
              pbar.update()

      net.train()

      if num_val == 0:
        return 0

      return val_loss/num_val
