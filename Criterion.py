import torch
import pdb

def entropy(pos, pos_hn=None):
    if pos_hn == None:
        loss = torch.log(pos +1e-12)
    else:
        loss = torch.log(pos*pos_hn +1e-8)
    loss = -loss
    loss = loss.mean()
    return loss
