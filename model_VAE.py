# 生成对抗网络模板
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from seq_improve_condition import conditioned_seq

# cuda setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#随便写的，没写完。先评价一下。
class EncoderLn(nn.Module):
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

class DecoderLn(nn.Module):
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
        
    def forward(self, hid_rep, class_def):
        input = np.concatenate((class_def, hid_rep))
        output = self.dec(input)
        return output
    
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, hid_layer, latent_dim, class_dim, output_dim, last_layer_activation=False):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.output_dim = output_dim
        self.last_layer_activation = last_layer_activation

        self.encoder = EncoderLn(input_dim, hid_layer, latent_dim, last_layer_activation)
        self.decoder = DecoderLn(latent_dim, class_dim, hid_layer, output_dim)

        self.padding = nn.ConstantPad1d(input_dim, -1)

    def forward(self, x):
        x = self.padding(x)
        latent_rep = self.encoder(x)
        conditioned_x = conditioned_seq(x, l=self.input_dim)
        dec_input = latent_rep + conditioned_x
        output = self.decoder(dec_input)
        return output

