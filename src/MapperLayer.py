from Omniglot import Omniglot
import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

import torch
from torch import nn
import torch.optim as optim
from torchvision.transforms.functional import adjust_contrast


import pickle
import random



class umwe2vae(nn.Module):
    def __init__(self,vae_model, in_dim=300, out_dim=128):
        super(umwe2vae, self).__init__()
        self.vae_model = vae_model
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = self.fc(x)
        # y = self.vae_model.decode(h)
        return h
        # here used to live post-processing
        #out = torch.zeros(y.shape)
        #for i in range(y.shape[0]):
        #    out[i] = adjust_contrast(y[i], contrast_factor=2.5)
        #return out

    def loss(self, x, alpha=1, beta=1):
        middle = x[:,:,1:-1,1:-1]
        ne     = x[:,:,0:-2,0:-2]
        n      = x[:,:,0:-2,1:-1]
        nw     = x[:,:,0:-2,2:]
        e      = x[:,:,1:-1,0:-2]
        w      = x[:,:,1:-1,2:]
        se     = x[:,:,2:,0:-2]
        s      = x[:,:,2:,1:-1]
        sw     = x[:,:,2:,2:]

        return alpha * torch.mean(sum([torch.abs(middle-ne),torch.abs(middle-n),torch.abs(middle-nw),torch.abs(middle-e),torch.abs(middle-w),torch.abs(middle-se),torch.abs(middle-s),torch.abs(middle-sw)]) / 8.) - beta * torch.mean(torch.abs(x-0.5))








class EmbeddingMapping(nn.Module):
    def __init__(self, device, embedding_vector_dim = 300, decoder_input_dim=128):
        super(EmbeddingMapping, self).__init__()
        self.device = device
        self.embedding_vector_dim = embedding_vector_dim
        self.decoder_input_dim = decoder_input_dim
        self.mapper_numlayer = 3

        self.linear_layers = []
        self.batch_norms = []
        for layer in range(0, self.mapper_numlayer-1):
            self.linear_layers.append(nn.Linear(embedding_vector_dim, embedding_vector_dim))
            self.batch_norms.append(nn.BatchNorm1d(embedding_vector_dim))

        # final layer
        self.linear_layers.append(nn.Linear(embedding_vector_dim, decoder_input_dim))
        self.batch_norms.append(nn.BatchNorm1d(decoder_input_dim))


        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)


        self.relu = nn.ReLU()
      
    def forward(self, embedding_vector):
        inp = embedding_vector
        for layer in range(self.mapper_numlayer):
            out = self.linear_layers[layer](inp)
            out = self.batch_norms[layer](out)
            out = self.relu(out)
            inp = out        
        return out



class MultilingualMapper(nn.Module):
    def __init__(self, device, embedding_vector_dim = 300, decoder_input_dim=128):
        super(MultilingualMapper, self).__init__()
        self.device = device
        self.embedding_vector_dim = embedding_vector_dim
        self.decoder_input_dim = decoder_input_dim
        self.mapper_numlayer = 3

        self.linear_layers = []
        self.batch_norms = []
        for layer in range(0, self.mapper_numlayer-1):
            self.linear_layers.append(nn.Linear(embedding_vector_dim, embedding_vector_dim))
            self.batch_norms.append(nn.BatchNorm1d(embedding_vector_dim))

        # final layer
        self.linear_layers.append(nn.Linear(embedding_vector_dim, decoder_input_dim))
        self.batch_norms.append(nn.BatchNorm1d(decoder_input_dim))


        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        self.relu = nn.ReLU()

        self.bce = nn.BCEWithLogitsLoss()
      
    def forward(self, embedding_vector):
        inp = embedding_vector
        for layer in range(self.mapper_numlayer):
            out = self.linear_layers[layer](inp)
            out = self.batch_norms[layer](out)
            out = self.relu(out)
            inp = out        
        return out

    def triplet_loss(self, sameWords_diffLangs, diffWords_sameLangs):
        return self.bce(sameWords_diffLangs, torch.ones(sameWords_diffLangs.shape).to(self.device)) + self.bce(diffWords_sameLangs, torch.zeros(diffWords_sameLangs.shape).to(self.device))


