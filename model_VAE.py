# 生成对抗网络模板
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from seq_improve_condition import conditioned_seq


#随便写的，没写完。先评价一下。
class EncoderLn(nn.Module):
    def __init__(self, input_dim, hid_layer, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(nn.Linear(input_dim, hid_layer[0]), nn.ReLU())#, nn.Dropout(p=0.3))
        
        for hidden_i in range(0, len(hid_layer) - 1):
            self.enc.append(nn.Linear(hid_layer[hidden_i], hid_layer[hidden_i + 1]))
            self.enc.append(nn.ReLU())

        self.mean_nn = nn.Linear(hid_layer[-1], latent_dim)
        self.log_var_nn = nn.Linear(hid_layer[-1], latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.enc(x)

        mean = self.mean_nn(x)
        log_var = self.log_var_nn(x)
        return mean, log_var

class DecoderLn(nn.Module):
    def __init__(self, latent_dim, hid_layer, output_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.dec = nn.Sequential(nn.Linear(latent_dim, hid_layer[0]), nn.ReLU())

        for hidden_i in range(0, len(hid_layer) - 1):
            self.dec.append(nn.Linear(hid_layer[hidden_i], hid_layer[hidden_i + 1]))
            self.dec.append(nn.ReLU())
        
        self.dec.append(nn.Linear(hid_layer[-1], output_dim))
        self.dec.append(nn.Sigmoid()) # Avoid blowing up
        
    def forward(self, x):
        x = self.dec(x)
        x = x.view(x.size(0), 1000, 21)

        return x

class Vaeln(nn.Module):
    def __init__(self, input_dim, hid_layer, latent_dim):
        super(Vaeln, self).__init__()

        self.encoder = EncoderLn(input_dim=input_dim, hid_layer=hid_layer, latent_dim=latent_dim)
        reverse_hid = self.hid_layer_reverse(hid_layer)
        self.decoder = DecoderLn(latent_dim=latent_dim, hid_layer=reverse_hid, output_dim=input_dim)   


    def hid_layer_reverse(self, hid_layer):
        reverse_hid_layer = []
        for i in range(len(hid_layer)):
            reverse_hid_layer.append(hid_layer[-i - 1])
        return reverse_hid_layer
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decoder(z)
        return recon_x, mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mean
            

    
# Copy from ChatGPT
class EncoderGPT(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(EncoderGPT, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # # Batch normalization layers
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        # Dropout
        self.drop1 = nn.Dropout1d(p=0.3)
        self.drop2 = nn.Dropout1d(p=0.3)
        self.drop3 = nn.Dropout1d(p=0.3)

        # Activation fuctions
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv1(x)
        # x = self.drop1(x)
        x = self.relu(x)

        
        x = self.conv2(x)
        # x = self.drop2(x)
        x = self.relu(x)

        x = self.conv3(x)
        # x = self.drop3(x)
        x = self.relu(x)

        # x = x.view(x.size(0), -1)
        x = torch.transpose(x, 1, 2)
        # (b, 128, 1000) -> (b, 1000, 128)
        
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        # (b, 1000, lat)
        
        return mean, logvar

class DecoderGPT(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(DecoderGPT, self).__init__()
        
        # Fully connected layer for mapping from latent space to hidden size
        self.fc = nn.Linear(latent_dim, 128)
        self.bn_fc = nn.BatchNorm1d(128)

         # Variational dropout layer
        self.variational_dropout = nn.Dropout(p=0.3)
        
        # Deconvolutional (transpose convolution) layers for upsampling
        self.deconv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv1d(32, output_size, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm1d(output_size)

        # Activation functions
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.prob_dist = nn.Softmax(dim=1) # column-wise softmax
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        # Input shape: (batch_size, latent_dim)
        
        # Fully connected layer
        # x = torch.cat([z, length.view(-1,1)], dim=-1)
        ### (b, 1000, lat) -> (b, 1000, 128)
        x = self.relu(self.fc(z))

        # x = self.bn_fc(x)
        # x = x.view(x.size(0), 128, 1000)
        x = torch.transpose(x, 1, 2)

        # Apply variational dropout
        # x = self.variational_dropout(x)
        
        # Apply deconvolutional layers
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)

        # x = self.bn_final(x)
        # x = self.prob_dist(x)
        # x = self.clear_padding(x, length)
        
        return x
    
    # def clear_padding(self, output_sequence, sequence_lengths):
    #     processed_output_sequence = output_sequence.clone()  # Create a copy of the original tensor
    
    #     # Clear the padding areas for each sequence in the batch
    #     for i in range(output_sequence.size(0)):
    #         seq_length = sequence_lengths[i]
    #         processed_output_sequence[i, :, seq_length:] = 0.0  # Set padding area to 0
            
    #     return processed_output_sequence
    
class VAEGPT(nn.Module):
    def __init__(self, input_size, latent_dim, output_size):
        super(VAEGPT, self).__init__()
        self.encoder = EncoderGPT(input_size, latent_dim)
        self.decoder = DecoderGPT(latent_dim, output_size)
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample_latent(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
    
    def sample_latent(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
