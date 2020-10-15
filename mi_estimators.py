import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils

####################
# MI LOWER BOUNDS ##
####################
def donsker_vardhan_lower_bound(scores,clip_scores = None):
    if clip_scores is None:
        clip_scores = scores
    out_true = scores.diag()
    out_false = utils.non_diag(clip_scores)
    return out_true.mean() - out_false.logsumexp(0) + math.log(out_false.shape[0])

def tuba_lower_bound(scores,log_baseline = None):
    if log_baseline is not None:
        scores = scores - log_baseline
    out_true = scores.diag()
    out_false = utils.non_diag(scores)
    joint_term = out_true.mean()
    marg_term = out_false.logsumexp(0).exp()/out_false.shape[0]
    return joint_term - marg_term + 1.0

def nwj_lower_bound(scores):
    return tuba_lower_bound(scores,1.0)

def js_fgan_lower_bound(scores):
    out_true = scores.diag()
    out_false = utils.non_diag(scores)
    return -1*F.softplus(-1*out_true).mean() - F.softplus(out_false).mean()


def smile_mi_lower_bound(scores,clamp = 5.0):
    scores_ = torch.clamp(scores,-1*clamp,clamp)
    return donsker_vardhan_lower_bound(scores,clip_scores = scores_)

class TUBAEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,scores,log_baseline = None):
        return tuba_lower_bound(scores,log_baseline = log_baseline)

class NWJEstimator(TUBAEstimator):
    def forward(self, scores):
        return nwj_lower_bound(scores)

class MINEEstimator(TUBAEstimator):
    def __init__(self, ema_rate = 0.99):
        super().__init__()
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        self.ema = nn.Parameter(torch.ones([1]),requires_grad = False).to(self.device)
        self.ema_rate = ema_rate

    def forward(self,scores):
        out_false = utils.non_diag(scores)
        new_baseline = (out_false.logsumexp(0) - math.log(out_false.shape[0])).exp()
        self.ema = nn.Parameter(self.ema * self.ema_rate + new_baseline * (1-self.ema_rate), requires_grad = False).to(self.device)
        return super().forward(scores, log_baseline = self.ema.log())

class JSNaiveEstimator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,scores):
        js = js_fgan_lower_bound(scores)
        mi = scores.diag().mean()
        return js + torch.detach(mi-js)
    
class JSEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores):
        js =  js_fgan_lower_bound(scores)
        mi = nwj_lower_bound(scores+1)
        return js + torch.detach(mi-js)

class SMILEEstimator(nn.Module):
    def __init__(self,clip = None):
        super().__init__()
        self.clip = clip

    def forward(self, scores):
        js = js_fgan_lower_bound(scores)
        
        if not self.clip:
            dv = donsker_vardhan_lower_bound(scores)
        else:
            dv = smile_mi_lower_bound(scores,clamp=self.clip)
        return js + torch.detach(dv-js)

def get_mi_estimator(name):
    if name.lower() == "nwj":
        return NWJEstimator
    elif name.lower() == "mine":
        return MINEEstimator
    elif name.lower() == "js":
        return JSEstimator
    elif name.lower() == "js_naive":
        return JSNaiveEstimator
    elif name.lower() == "smile":
        return SMILEEstimator
    else:
        raise NotImplementedError("MI estimator with name {} is not implemented".format(name))
