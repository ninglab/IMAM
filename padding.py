import numpy as np
import pdb

def padding_autogressiv(batch_data, max_len, num_item, num_pred=1, oracle=False):
    
    batch_train = []
    batch_test  = []

    for i in range(len(batch_data)):
        single_data = batch_data[i][-max_len:]
        batch_test.append([single_data[-1]])
        if not oracle:
            single_data = single_data[:-num_pred]
            batch_train.append( [0] * (max_len-1-len(single_data)) + single_data)
        else:
            batch_train.append( [0] * (max_len-len(single_data)) + single_data)

    return np.array(batch_train), np.array(batch_test)

def padding(batch_data, num_item, global_max_len, model):
    #index 0 is used as padding
    #the item index start from 1
    if model == 'oracle':
        max_len = global_max_len
        oracle = True
    else:
        max_len = global_max_len + 1
        oracle = False
    batch_train = []
    batch_test  = []

    mat, pred = padding_autogressiv(batch_data, max_len, num_item, oracle=oracle)

    return mat, pred
