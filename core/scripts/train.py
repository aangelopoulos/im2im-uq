import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import argparse
import logging
import wandb

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from core.scripts.eval import eval_net

from torch.utils.data import DataLoader, random_split
import core.utils as utils
import pdb
import dill as pkl

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
    # If we're loading from a checkpoint, do so.
    if load_from_checkpoint:
      checkpoint_final_path = checkpoint_dir + f'/CP_epoch{epochs}.pth'
      if os.path.exists(checkpoint_final_path):
        with open(checkpoint_final_path, 'rb') as f:
          net = pkl.load(f)
        print(f"Model loaded from checkpoint {checkpoint_final_path}")
        return
    
    # Otherwise, train the model
    global_step = 0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    net.to(device=device)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    # WandB magic
    try:
      wandb.watch(net, log_freq = 100)
    except:
      wandb.init(config=config)
      wandb.watch(net, log_freq = 100)

    for epoch in range(epochs):
        net.train()
        print('epoch ' + str(epoch+1) + '\n')
        epoch_loss = 0
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
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            global_step += 1

        wandb.log({"iter":global_step, "train_loss":epoch_loss/len(train_loader)})

        with torch.no_grad():

          if (epoch+1) % validate_every == 0:
              # validation
              val_loss = eval_net(net, val_loader, device)
              wandb.log({"iter":global_step, "val_loss":val_loss})
              # TODO: Figure out how to log this in a general way.
              try:
                wandb.log({"iter":global_step, "examples_input": [wandb.Image((255 * val_dataset[img_idx][0].squeeze()).numpy().astype(np.uint8)) for img_idx in range(5)]})
                examples_output = [(255 * net(val_dataset[img_idx][0].unsqueeze(0).to(device)).cpu().squeeze().permute(1,2,0)).numpy() for img_idx in range(5)]
                examples_output = [np.maximum(0,np.minimum(example, 255)) for example in examples_output]
                examples_lower_quantile = [wandb.Image(example[:,:,0]) for example in examples_output]
                examples_prediction = [wandb.Image(example[:,:,1]) for example in examples_output]
                examples_upper_quantile = [wandb.Image(example[:,:,2]) for example in examples_output]
                wandb.log({"iter":global_step, "Lower quantile": examples_lower_quantile})
                wandb.log({"iter":global_step, "Predictions": examples_prediction})
                wandb.log({"iter":global_step, "Upper quantile": examples_upper_quantile})
                wandb.log({"iter":global_step, "Ground truth": [wandb.Image((255 * val_dataset[img_idx][1].squeeze()).numpy().astype(np.uint8)) for img_idx in range(5)]})
              except:
                print("Failed logging images.")
              #scheduler.step(val_loss)

          if (epoch+1) % checkpoint_every == 0:
              print('saving checkpoint')
              if checkpoint_dir != None:
                  try:
                      os.mkdir(checkpoint_dir)
                      logging.info('Created checkpoint directory')
                  except OSError:
                      pass
                  with open(checkpoint_dir + f'/CP_epoch{epoch + 1}.pth', 'wb') as handle:
                    print("TODO: FIX ERROR WITH MULTIPROCESSING; MODEL NOT THREADSAFE")
                    net.eval()
                    _net = pkl.dumps(net)
                    pkl.dump(_net, handle, protocol=pkl.HIGHEST_PROTOCOL)
                    net.train()

                  logging.info(f'Checkpoint {epoch + 1} saved !')
        net.eval()
