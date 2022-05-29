import numpy as np
import random
from utils import *
from padding import *
import pdb

class Dataloader():
    def __init__(self, args):

        if args.data == 'gowalla':
            path = 'datasets/gowalla/'
        elif args.data == 'lastfm':
            path = 'datasets/lastfm/'
        elif args.data == 'yoochoose1_4':
            path = 'datasets/yoochoose1_4/'
        elif args.data == 'yoochoose1_64':
            path = 'datasets/yoochoose1_64/'
        elif args.data == 'diginetica':
            path = 'datasets/diginetica/'
        elif args.data == 'nowplaying':
            path = 'datasets/nowplaying/'
        elif args.data == 'tmall':
            path = 'datasets/tmall/'

        self.train_set, self.test_set, self.num_items = self.load_train_test(path, args.isTrain)

        self.shift = args.min_item_idx
        self.max_len = args.max_len
        self.model = args.model

        self.id_list = None
        self.device  = None 

    def load_train_test(self, path, isTrain):

        def load(path, seqs_f):
            seqs = []
            with open(path+seqs_f, 'r') as f:
                for line in f.readlines():
                    line = list(map(int, line.rstrip('\n').split(',')))
                    for i in range(1, len(line)):
                        seqs.append(line[:i+1])
            return seqs

        if isTrain:
            train_set = load(path, 'train_valid.txt')
            test_set = load(path, 'test_valid.txt')
            with open(path+'num_items_valid.txt', 'r') as f:
                num_items = int(f.readlines()[0].rstrip('\n'))
            
        else:
            train_set = load(path, 'train.txt')
            test_set = load(path, 'test.txt')
            with open(path+'num_items.txt', 'r') as f:
                num_items = int(f.readlines()[0].rstrip('\n'))

        return train_set, test_set, num_items

    def set_id_list(self, isTrain, isEval):

        len_data = len(self.train_set) if not isEval else len(self.test_set)

        if isEval:
            self.id_list = list(range(len_data))
            return

        if (not self.id_list) or (len(self.id_list) != len_data):
            self.id_list = list(range(len_data))

        random.shuffle(self.id_list)
        return

    def get_test_set(self, args):
        #test on the last item of testing sessions
        test_set = [[sess[-1]] for sess in self.test_set]
        return test_set

    def get_data_len(self, isEval):
        len_data = len(self.train_set) if not isEval else len(self.test_set)
        return len_data

    def generate_batch(self, num_batch, batch_size, num_epoch, isTrain, isEval):

        data = self.train_set if not isEval else self.test_set

        start = batch_size * num_batch
        end   = min(batch_size * (num_batch + 1), len(data))

        batch_id_list = self.id_list[start:end]
        batch_data = [data[i] for i in batch_id_list]

        mat, pred = padding(batch_data, self.num_items, self.max_len, self.model)

        pred = torch.from_numpy(pred).type(torch.LongTensor).to(self.device)
        mat  = torch.from_numpy(mat).type(torch.LongTensor).to(self.device)

        return mat, pred
    
