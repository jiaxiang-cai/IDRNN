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

        self.enc = nn.Sequential(nn.Linear(input_dim, hid_layer[0]), nn.ReLU())
        
        for hidden_i in range(0, len(hid_layer) - 1):
            self.enc.append(nn.Linear(hid_layer[hidden_i], hid_layer[hidden_i + 1]))
            self.enc.append(nn.ReLU())

        self.mean_nn = nn.Linear(hid_layer[-1], latent_dim)
        self.log_var_nn = nn.Linear(hid_layer[-1], latent_dim)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
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
        self.dec.append(nn.Softmax())
        
    def forward(self, x):
        x = self.dec(x)
        x = torch.transpose(x, 1, 2)

        return x
    
# class ConditionalVAE(nn.Module):
#     def __init__(self, input_dim, hid_layer, latent_dim, class_dim, output_dim, last_layer_activation=False):
#         super().__init__()

#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.class_dim = class_dim
#         self.output_dim = output_dim
#         self.last_layer_activation = last_layer_activation

#         self.encoder = EncoderLn(input_dim, hid_layer, latent_dim, last_layer_activation)
#         self.decoder = DecoderLn(latent_dim, class_dim, hid_layer, output_dim)

#         self.padding = nn.ConstantPad1d(input_dim, -1)

#     def forward(self, x):
#         x = self.padding(x)
#         latent_rep = self.encoder(x)
#         conditioned_x = conditioned_seq(x, l=self.input_dim)
#         dec_input = latent_rep + conditioned_x
#         output = self.decoder(dec_input)
#         return output

# Copy from ChatGPT

class EncoderGPT(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(EncoderGPT, self).__init__()
        
        # Convolutional layers for local feature extraction
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers for mapping to latent space
        self.fc_mean = nn.Linear(128 * 1000, latent_dim)
        self.fc_logvar = nn.Linear(128 * 1000, latent_dim)
        
    def forward(self, x):
        # Input shape: (batch_size, input_size, sequence_length)
        
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Calculate mean and log variance for latent distribution
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar

# # Example usage
# input_size = 20  # Number of amino acids
# latent_dim = 3
# sequence_length = 1000  # Padded sequence length

# # Create an instance of the Encoder
# encoder = Encoder(input_size, latent_dim)

# # Generate a random input batch
# batch_size = 32
# input_data = torch.randn(batch_size, input_size, sequence_length)

# # Forward pass through the encoder
# mean, logvar = encoder(input_data)

# print("Mean shape:", mean.shape)
# print("Logvar shape:", logvar.shape)

class DecoderGPT(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(DecoderGPT, self).__init__()
        
        # Fully connected layer for mapping from latent space to hidden size
        self.fc = nn.Linear(latent_dim, 128 * 1000)
        
        # Deconvolutional (transpose convolution) layers for upsampling
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, output_size, kernel_size=3, padding=1)
        
    def forward(self, z):
        # Input shape: (batch_size, latent_dim)
        
        # Fully connected layer
        x = self.fc(z)
        x = x.view(x.size(0), 128, 1000)
        
        # Apply deconvolutional layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = torch.sigmoid(self.deconv3(x))
        
        return x
    
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