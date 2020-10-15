import os
import sys
import math
import itertools
import glob
import re

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as dist

def perm_axis(x,dim = 0):
    b = x.shape[dim]
    perm = torch.randperm(b, device = x.device)
    return torch.index_select(x,dim,perm)

def non_diag(x):
    assert x.shape[0] == x.shape[1]
    tril_indices = torch.tril_indices(x.shape[0],x.shape[1],-1)
    triu_indices = torch.triu_indices(x.shape[0],x.shape[1],1)
    indices = torch.cat([tril_indices,triu_indices],dim=-1)
    return x[indices[0],indices[1]]

def get_batch_pairs(x):
    b = x.shape[0]
    comb = list(zip(*itertools.combinations(range(b),2)))
    x1 = torch.index_select(x,0,torch.tensor(comb[0],dtype = torch.long,device = x.device))
    x2 = torch.index_select(x,0,torch.tensor(comb[1],dtype = torch.long,device = x.device))
    return x1,x2

def save_checkpoints(checkpoint, name, max_chkps = 5): 
    #Find all checkpoints in the folder
    dirname = os.path.dirname(name)
    checkpoint_files = glob.glob(os.path.join(dirname,"*.pth"))
    if len(checkpoint_files) >= max_chkps:
        checkpoint_files.sort(key = lambda x: int(re.findall(r"\d+",x)[-1]))
        os.remove(checkpoint_files[0])
    torch.save(checkpoint, name)

def get_latest_checkpoint_file(dir):
    checkpoint_files = glob.glob(os.path.join(dir,"*.pth"))
    checkpoint_files.sort(key = lambda x: int(re.findall(r"\d+",x)[-1]),reverse = True)
    if checkpoint_files:
        return checkpoint_files[0]
    else:
        return None

def log_gradients(module, summary_writer, global_step = None):
    for tag, value in module.named_parameters():
        tag = tag.replace(".","/")
        if value.grad is not None:
            summary_writer.add_scalar(tag+"/grad_norm", value.grad.norm(), global_step = global_step)
            summary_writer.add_histogram(tag+"/grad", value.grad, global_step = global_step)

def make_critic_input(x,y):
    batch_size,dims = x.shape
    x = x[None].repeat(batch_size,1,1).reshape([-1,dims])
    y = y[:,None].repeat(1,batch_size,1).reshape([-1,dims])
    return torch.cat([x,y],1)

def get_correlated_gaussian(rho,batch_size,dims,cube=False):
    x,eps = torch.chunk(dist.Normal(0.0,1.0).sample([batch_size,2*dims]),2,1)
    y = rho*x + torch.sqrt(1-torch.pow(rho,2))*eps
    if cube:
        y = torch.pow(y,3)
    return x,y

def rho_to_mi(dims,rho):
    return -0.5 * dims * torch.log(1-torch.pow(rho,2))
    
def mi_to_rho(dims,mi):
    return torch.sqrt(1 - torch.exp(-2.0/dims*mi))

def mi_scheduler(niters):
    mi_list = torch.arange(2,21,2, dtype = torch.float)
    repeat_factor = niters//len(mi_list)
    mi_list = mi_list.repeat_interleave(repeat_factor)
    return mi_list
