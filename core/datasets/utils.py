#Useful data utils
import numpy as np

def normalize(x, type, per_pixel, input_output):
    if type == 'standard':
        # code
        if per_pixel:
            mean_val = x.mean(dim=0)[:, None, :, :]
            std_val = x.std(dim=0)[:, None, :, :]
        else:
            mean_val = x.mean()
            std_val = x.std()
        params = {'mean_'+input_output: mean_val, 'std_'+input_output: std_val}
        x = (x - mean_val) / std_val

    elif type == 'min-max':
        if per_pixel:
            max_val = x.max(dim=0)[0][:, None, :, :]
            min_val = x.min(dim=0)[0][:, None, :, :]
        else:
            max_val = x.max()
            min_val = x.min()
        params = {'max_'+input_output: max_val, 'min_'+input_output: min_val}
        x = (x - min_val) / (max_val - min_val)

    else:
        raise NotImplementedError

    return x, params
