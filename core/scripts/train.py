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
from core.networks.unet import UNet
from core.networks.unet import WNet

from torch.utils.data import DataLoader, random_split
from core.get_dataset import get_pytorch_dataset
import core.utils as utils
import pdb
from datetime import datetime
import pickle as pkl

def train_net(net,
              train_dataset,
              val_dataset,
              device,
              epochs,
              batch_size,
              lr,
              dir_checkpoint,
              dir_output,
              save_cp):

    global_step = 0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    net.to(device=device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # if net.n_classes > 1:
    #     print('performing classification with integer pixel values as classes')
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     print('performing regression')
    #     criterion = nn.MSELoss()

    for epoch in range(epochs):
        net.train()
        print('epoch ' + str(epoch+1) + '\n')
        epoch_loss = 0
        for batch in train_loader:
            labels = batch[-1]
            x = tuple([batch[i] for i in range(len(batch)-1)])
            x = [x[i].to(device=device, dtype=torch.float32) for i in range(len(x))]
            labels = labels.to(device=device)

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

        if (epoch+1) % 10 == 0:
            # validation
            val_loss = eval_net(net, val_loader, device)
            wandb.log({"iter":global_step, "train_loss":val_loss})
            #scheduler.step(val_score)

        if (epoch+1) % 10 == 0:
            print('saving checkpoint')
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net,
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
        net.eval()
