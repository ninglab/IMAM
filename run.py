import argparse
from DataLoader import Dataloader
from utils import train, evaluation
import pdb
import os
from Criterion import entropy
import torch
import logging
from time import time
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', type=int, default=0, help='optimizer, 0: Adam, 1: AdamW, 2: Adagrad')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4096)
    #n_epoch=60 for diginetica
    #n_epoch=20 for nowplaying
    #n_epoch=30 for tmall
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--data', type=str, default='yoochoose1_64')
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--min_item_idx', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=11)
    parser.add_argument('--dropout_rate', type=int, default=0.5)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--model', type=str, default='FF')

    args = parser.parse_args()

    data_loader = Dataloader(args)
    data_loader.device = device

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    criterion = entropy

    from models.MAM import MAM
    model = MAM(data_loader.num_items, device, args).to(device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers

    train(args, model, data_loader, criterion, logger)
    
    print('END')
