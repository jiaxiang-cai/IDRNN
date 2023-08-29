# 生成对抗网络模板
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from seq_improve_condition import conditioned_seq
    
# Copy from ChatGPT
class EncoderCNN(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(EncoderCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # # Batch normalization layers
        # self.bn1 = nn.BatchNorm1d(32)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)

        # # Dropout
        # self.drop1 = nn.Dropout1d(p=0.3)
        # self.drop2 = nn.Dropout1d(p=0.3)
        # self.drop3 = nn.Dropout1d(p=0.3)

        # Activation fuctions
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        # input x of shape (batch, 1000, 21) -> (batch, 21,  1000)
        x = x.transpose(1,2)
        x = self.conv1(x)
        # x = self.drop1(x)
        # x = self.relu(x)

        
        x = self.conv2(x)
        # x = self.drop2(x)
        # x = self.relu(x)

        x = self.conv3(x)
        # x = self.drop3(x)
        # x = self.relu(x)

        x = x.transpose(1, 2)
        # (b, 128, 1000) -> (b, 1000, 64)
        
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        # (b, 1000, lat)
        
        return mean, logvar

class DecoderCNN(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(DecoderCNN, self).__init__()
        
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
        ### (b, 1000, lat) -> (b, 1000, 64)
        x = self.relu(self.fc(z))

        # x = self.bn_fc(x)
        # (b, 1000, 64) -> (b, 64, 1000)
        x = torch.transpose(x, 1, 2)

        # Apply variational dropout
        # x = self.variational_dropout(x)
        
        # Apply deconvolutional layers
        x = self.deconv1(x)
        # x = self.relu(x)
        x = self.deconv2(x)
        # x = self.relu(x)
        x = self.deconv3(x)
        # x = self.relu(x)

        # x = self.bn_final(x)
        # x = self.prob_dist(x)
        
        return x.transpose(1, 2)
    # return (b, 1000, 21)

    
class VAECNN(nn.Module):
    def __init__(self, input_size, latent_dim, output_size):
        super(VAECNN, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = EncoderCNN(input_size, latent_dim)
        self.decoder = DecoderCNN(latent_dim, output_size)
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sample_latent(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar
    
    def sample_latent(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def generate_sequence(self, num_seq, seq_len):

        z = torch.randn(num_seq, seq_len, self.latent_dim)

        seq = self.decoder(z)
        seq = seq.permute(0, 2, 1)
        seq = nn.Softmax(dim=1)(seq)
        return seq
