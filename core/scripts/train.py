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
              checkpoint_dir,
              checkpoint_every,
              validate_every,
              config=None): # config not normally needed due to wandb

    global_step = 0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
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
        for batch in train_loader:
            labels = batch[-1]
            x = tuple([batch[i] for i in range(len(batch)-1)])
            x = [x[i].to(device=device, dtype=torch.float32) for i in range(len(x))]
            labels = labels.to(device=device)

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

        if (epoch+1) % validate_every == 0:
            # validation
            val_loss = eval_net(net, val_loader, device)
            wandb.log({"iter":global_step, "val_loss":val_loss})
            # TODO: Figure out how to log this in a general way.
            #wandb.log({"examples": [wandb.Image(i) for i in Images]})
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
                  _net = pkl.dumps(net)
                  pkl.dump(_net, handle, protocol=pkl.HIGHEST_PROTOCOL)

                logging.info(f'Checkpoint {epoch + 1} saved !')
        net.eval()
