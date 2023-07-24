# 生成对抗网络模板
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

# cuda setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#随便写的，没写完。先评价一下。
class EncoderLn(nn.module):
    def __init__(self, input_dim, hid_layer, latent_dim, last_layer_activation=False):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Linear(input_dim, hid_layer[0]), nn.ReLU()
        )
        for hidden_i in range(1, len(hid_layer) - 1):
            self.enc.append(nn.Linear(hid_layer[hidden_i], hid_layer[hidden_i + 1]))
            self.enc.append(nn.BatchNorm1d(hid_layer[hidden_i + 1]))
            self.enc.append(nn.ReLU())

        self.enc.append(nn.Linear(hid_layer[-1], latent_dim))
        self.enc.append(nn.BatchNorm1d(latent_dim))
        if last_layer_activation:
            self.enc.append(nn.ReLU())

    def forward(self, x):
        out = self.enc(x)
        return x, out

class DecoderLn(nn.module):
    def __init__(self, latent_dim, class_dim, hid_layer, output_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.output_dim = output_dim

        self.dec = nn.Sequential(
            nn.Linear(class_dim + latent_dim, hid_layer[0]), nn.ReLU()
        )
        for hidden_i in range(1, len(hid_layer) - 1):
            self.dec.append(nn.Linear(hid_layer[hidden_i], hid_layer[hidden_i + 1]))
            self.dec.append(nn.BatchNorm1d(hid_layer[hidden_i + 1]))
            self.dec.append(nn.ReLU())
        
        self.dec.append(nn.Linear(hid_layer[-1], output_dim))
        
    def forward(self, hid_rep, x):
        input = np.concatenate((x, hid_rep))
        output = self.dec(input)
        return output
    
class ConditionalVAE(nn.module):
    def __init__(self, x):
        pass

