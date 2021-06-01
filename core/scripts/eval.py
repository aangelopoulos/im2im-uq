import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def eval_net(net, loader, device):
    net.eval()
    #label_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    if n_val == 0:
        print("No val points, returning 0")
        return 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            x, labels = batch
            if net.n_classes > 1:
                # classification
                labels = (labels * 255.0).to(device=device, dtype=torch.long)[:, 0, :, :]  # get rid of extra dim for loss
            else:
                # regression
                labels = labels.to(device=device, dtype=torch.float32)
            x = x.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                labels_pred = net(x)

            # if net.n_classes > 1:
            #     tot += F.cross_entropy(labels_pred, labels).item()
            # else:
            #     tot += F.mse_loss(labels_pred, labels).item()

            tot += net.loss_fn(labels_pred, labels).item()
            pbar.update()

    net.train()
    return tot / n_val
