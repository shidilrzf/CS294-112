import numpy as np
import torch.nn as nn
from torch.nn import init


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


def normalize(x, mu=0.0, std=1.0):
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return (x + mu) * (std + 1e-8)