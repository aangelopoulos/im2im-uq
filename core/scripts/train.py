import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import argparse
import copy
import logging
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from core.scripts.eval import get_images, eval_net

from torch.utils.data import DataLoader, random_split
import core.utils as utils
import pdb
import dill as pkl

class DataParallelPassthrough(nn.DataParallel):
  def __getattr__(self, name):
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.module, name)

def run_validation(net,
                   val_loader,
                   val_dataset,
                   device,
                   global_step,
                   epoch,
                   config):
  with torch.no_grad():
    net.eval()
    # Plot images 
    try:
      # Get the prediction sets and properly organize them 
      examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth, examples_ll, examples_ul, results_list = get_images(net,
                                                                                                                                      val_dataset,
                                                                                                                                      device,
                                                                                                                                      list(range(config['num_validation_images'])),
                                                                                                                                      config)

      # Log everything
      wandb.log({"epoch": epoch, "iter":global_step, "examples_input": examples_input})
      wandb.log({"epoch": epoch, "iter":global_step, "Lower edge": examples_lower_edge})
      wandb.log({"epoch": epoch, "iter":global_step, "Predictions": examples_prediction})
      wandb.log({"epoch": epoch, "iter":global_step, "Upper edge": examples_upper_edge})
      wandb.log({"epoch": epoch, "iter":global_step, "Ground truth": examples_ground_truth})
      wandb.log({"epoch": epoch, "iter":global_step, "Lower length": examples_ll})
      wandb.log({"epoch": epoch, "iter":global_step, "Upper length": examples_ul})
    except:
      print("Failed logging images.")
    val_loss = eval_net(net, val_loader, device)
    wandb.log({"epoch": epoch, "iter":global_step, "val_loss":val_loss})
    print(f"Val loss: {val_loss}")
  net.train()

def train_net(net,
              train_dataset,
              val_dataset,
              device,
              epochs,
              batch_size,
              lr,
              load_from_checkpoint,
              checkpoint_dir,
              checkpoint_every,
              validate_every,
              config=None): # config not normally needed due to wandb

    # MODEL LOADING CODE
    starting_epoch = 0 # will change if loading from checkpoint
    if config == None:
      config = wandb.config
    # If we're loading from a checkpoint, do so.
    if load_from_checkpoint:
      checkpoint_final_path = checkpoint_dir + f'/CP_epoch{epochs}_' + config['dataset'] + "_" + config['uncertainty_type'] + "_" + str(config['batch_size']) + "_" + str(config['lr']) + "_" + config['input_normalization'] + "_" + config['output_normalization'].replace('.','_') + '.pth'
      if os.path.exists(checkpoint_final_path):
        try:
          net = torch.load(checkpoint_final_path)
          net.eval()
          print(f"Model loaded from checkpoint {checkpoint_final_path}")
          return net
        except:
          print(f"Final model cannot be loaded from checkpoint {checkpoint_final_path}. Training now, for {epochs} epochs.")
      else:
        print(f"Final model cannot be loaded from checkpoint {checkpoint_final_path}. Training now, for {epochs} epochs.")
        for e in reversed(range(epochs)):
          checkpoint_intermediate_path = checkpoint_dir + f'/CP_epoch{e}_' + config['dataset'] + "_" + config['uncertainty_type'] + "_" + str(config['batch_size']) + "_" + str(config['lr']) + "_" + config['input_normalization'] + "_" + config['output_normalization'].replace('.','_') + '.pth'
          if os.path.exists(checkpoint_intermediate_path):
            net = torch.load(checkpoint_intermediate_path)
            starting_epoch = e
            print(f"Starting from epoch {e}.")
            break
    
    # Otherwise, train the model
    global_step = 0

    try:
      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except:
      train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Train loader!")

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    net = net.to(device=device)
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
    #  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      net = DataParallelPassthrough(net) # If you only want to train on two GPUs, add device_ids=[0,1]

    print("DataParallel!")
    net = net.to(device=device)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    # WandB magic
    print("WandB Magic!")
    if starting_epoch == 0:
      try:
        wandb.watch(net, log_freq = 100)
      except:
        wandb.init(config=config)
        wandb.watch(net, log_freq = 100)

    # If you want, you can run an initial validation to see how bad a random net is
    #run_validation(net,
    #               val_loader,
    #               val_dataset,
    #               device,
    #               global_step,
    #               starting_epoch,
    #               config)

    print("Start Training!")
    for epoch in range(starting_epoch,epochs):
        net = net.to(device)
        net.train()
        print('epoch ' + str(epoch+1) + '\n')
        epoch_loss = 0
        num_examples = 0
        for batch in tqdm(train_loader):
            labels = batch[-1].to(device=device)
            x = tuple([batch[i].to(device=device, dtype=torch.float32) for i in range(len(batch)-1)])

            # Predict
            labels_pred = net(*x) # Unpack tuple
            loss = net.loss_fn(labels_pred, labels)
            loss.retain_grad()
            epoch_loss += loss.item()

            # Take gradient step 
            optimizer.zero_grad()
            loss.backward()
            loss.retain_grad()
            #nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            global_step += 1
            num_examples += labels.shape[0]

        wandb.log({"iter":global_step, "train_loss":epoch_loss/max(num_examples,1)})

        with torch.no_grad():
          #net.load_state_dict(net.state_dict())

          if (epoch) % validate_every == 0:
              # validation
              run_validation(net,
                             val_loader,
                             val_dataset,
                             device,
                             global_step,
                             epoch,
                             config)
              #scheduler.step(val_loss)

          if (epoch+1) % checkpoint_every == 0:
              print('saving checkpoint')
              if checkpoint_dir != None:
                  try:
                      os.makedirs(checkpoint_dir,exist_ok=True)
                      logging.info('Created checkpoint directory')
                  except OSError:
                      pass
                  checkpoint_fname = checkpoint_dir + f'/CP_epoch{epoch + 1}_' + config['dataset'] + "_" + config['uncertainty_type'] + "_" + str(config['batch_size']) + "_" + str(config['lr']) + "_" + config['input_normalization'] + "_" + config['output_normalization'].replace('.','_') + '.pth'
                  torch.save(net.cpu().module, checkpoint_fname)
                  #torch.save(net.module, checkpoint_fname) #Without wandb, you don't need the .module call

                  logging.info(f'Checkpoint {epoch + 1} saved !')
        net.eval()
    return net#.module
