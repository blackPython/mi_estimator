import math
import os
import sys

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

class LinearModel(nn.Module):
    def __init__(self,dims,nf = 512,nlayers = 3):
        super().__init__()
        self.dims = dims
        self.nf = nf
        
        self.in_layer = nn.Sequential(
                        nn.Linear(dims,nf),
                        nn.ReLU(),
                        )
        
        hidden_layers = []
        for _ in range(nlayers-2):
            hidden_layers.append(
                nn.Sequential(
                    nn.Linear(nf,nf),
                    nn.ReLU()
                )
            )
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.out_layer = nn.Linear(nf,1)
        
    def forward(self,x):
        return self.out_layer(self.hidden_layers(self.in_layer(x)))

class ASKLLayer(nn.Module):
    def __init__(self,in_dims,out_dims,stationary=False):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.stationary = stationary
        uniform_dist = dist.Uniform(0,2*np.pi)
        
        if not stationary:
            self.omega1 = nn.Parameter(torch.empty(in_dims,out_dims),requires_grad=True)
            self.omega2 = nn.Parameter(torch.empty(in_dims,out_dims),requires_grad=True)
            self.b1 = nn.Parameter(uniform_dist.sample((1,out_dims)),requires_grad=False)
            self.b2 = nn.Parameter(uniform_dist.sample((1,out_dims)),requires_grad=False)
        else:
            self.omega = nn.Parameter(torch.empty(in_dims,out_dims),requires_grad=True)
            self.b = nn.Parameter(uniform_dist.sample((1,out_dims)),requires_grad=False)
    
    def init(self,std1,std2):
        if not self.stationary:
            nn.init.normal_(self.omega1,0,std1)
            nn.init.normal_(self.omega2,0,std2)
        else:
            nn.init.normal_(self.omega,0,std1)

    def forward(self,x):
        if not self.stationary:
            phi = (torch.cos(x.mm(self.omega1)+self.b1)+torch.cos(x.mm(self.omega2)+self.b2))/math.sqrt(2*self.out_dims)
        else:
            phi = torch.cos(x.mm(self.omega)+self.b)*math.sqrt(2.0/self.out_dims)
        
        return phi

class MultiASKLLayers(nn.Module):
    def __init__(self,in_dims,out_dims,nf=256,n_layers=2):
        super().__init__()
        self.nf = nf
        self.n_layers = n_layers
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.askl_layers = nn.ModuleList([ASKLLayer(in_dims if i == 0 else nf, nf if i < (n_layers-1)
                                                    else out_dims) for i in range(n_layers)])
    
    def init(self,std1,std2):
        for layer,s1,s2 in zip(self.askl_layers,std1,std2):
            layer.init(s1,s2)
    
    def forward(self,x):
        hidden_values = [x]
        hidden = x
        for layer in self.askl_layers:
            hidden = layer(hidden)
            hidden_values.append(hidden)
        return hidden,hidden_values
class GhimreCritic(nn.Module):
    def __init__(self,in_dims,nf=256,nlayers=2,nsamples=256):
        super().__init__()
        self.nf = nf
        self.nlayers = nlayers
        self.in_dims = in_dims
        self.nsamples = nsamples
        embed_layers = []
        for i in range(nlayers):
            embed_layers.append(nn.Linear(in_dims if i==0 else nf,nf))
            embed_layers.append(nn.ReLU())
        self.embed_layers = nn.Sequential(*embed_layers)
        self.mu = nn.Parameter(torch.empty(nf),requires_grad = True)
        self.logvar = nn.Parameter(torch.empty(nf),requires_grad = True)
        nn.init.normal_(self.mu,0,0.1)
        nn.init.normal_(self.logvar,0,0.1)
    def forward(self,x):
        device = x.device
        embed_mapping = self.embed_layers(x)
        eps = dist.Normal(0,0.1).sample((self.nsamples,self.nf)).to(device)
        std = (self.logvar * 0.5).exp()
        w = self.mu[None] + std[None] * eps #Size nsamplesxnf
        fx = torch.mean(embed_mapping.mm(w.T),1)
        middle_term = self.mu[None].T.mm(self.mu[None]) + torch.diag(self.logvar.exp())
        K = embed_mapping.mm(middle_term).mm(embed_mapping.T)
        return fx,K
