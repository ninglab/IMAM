import numpy as np
import torch
import pdb
from scipy.special import softmax
from time import time
from eval_metrics import precision_at_k, recall_at_k, mapk, ndcg_k, mrr_k

ID2OPT = {0:torch.optim.Adam, 1:torch.optim.AdamW, 2:torch.optim.Adagrad}

def set_optimizer(args, model):
    optimizer = ID2OPT[args.optim](model.parameters(), lr=args.lr, weight_decay=args.l2)
    return optimizer

def train(args, model, data_loader, criterion, logger):

    optimizer = set_optimizer(args, model)
    
    lambda1 = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    len_data = data_loader.get_data_len(isEval=0)
    n_batch = len_data // args.batch_size

    epoch_loss = 0
    for epoch in range(args.n_epoch):

        t1 = time()

        data_loader.set_id_list(isTrain=args.isTrain, isEval=0)

        for batch in range(n_batch):
            mat, pred = data_loader.generate_batch(batch, args.batch_size, epoch, isTrain=args.isTrain, isEval=0)

            pos = model(mat, pred, isEval=0)
            loss = criterion(pos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #zero padding
            with torch.no_grad():
                zeros = torch.zeros(args.dim).to(data_loader.device)
                for i in range(args.min_item_idx):
                    model.item_emb.weight[i].copy_(zeros)

        epoch_loss /= n_batch
        scheduler.step()

        t2 = time()

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        k = 5
        if (epoch + 1) % k == 0:

            #set the model to eval mode
            model.eval()
            with torch.no_grad():
                recall, ndcg, mrr = evaluation(args, model, data_loader, topk=20)

            logger.info('rec: %.4f,%.4f,%.4f,%.4f' % (recall[0], recall[1], recall[2], recall[3]))
            logger.info('NDCG: %.4f,%.4f,%.4f,%.4f'% (ndcg[0], ndcg[1], ndcg[2], ndcg[3]))
            logger.info('MRR: %.4f,%.4f,%.4f,%.4f' % (mrr[0], mrr[1], mrr[2], mrr[3]))

            logger.info("Evaluation time:{}".format(time() - t2))
            model.train()

    logger.info("\n")
    logger.info("\n")

    return

def evaluation(args, model, data_loader, topk=50):

    len_data = data_loader.get_data_len(1)
    if len_data % args.batch_size == 0:
        n_batch  = len_data // args.batch_size
    else:
        n_batch  = len_data // args.batch_size + 1
    test_set = data_loader.get_test_set(args)
    data_loader.set_id_list(isTrain=args.isTrain, isEval=1)
    mask_exist = -1e4
    alpha_list = None

    sim_truth = None
    sim_mean = None 
    feats_sim_truth = None 
    feats_sim_mean = None 
    multi_attn = None 
    length = None

    for batch in range(n_batch):
        mat, pred = data_loader.generate_batch(batch, args.batch_size, num_epoch=None, isTrain=args.isTrain, isEval=1)
        rating_pred =  model(mat, pred, isEval=1)
        rating_pred = rating_pred.cpu().data.numpy().copy()

        #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array/23734295#23734295
        ind = np.argpartition(rating_pred, -topk)[:, -topk:]
        
        #https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops/20104162#20104162
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        #switch the inds to the right order
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batch == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    recall, ndcg, mrr = [], [], []
    for k in [5, 10, 15, 20]:

        recall.append(recall_at_k(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))
        mrr.append(mrr_k(test_set, pred_list, k))

    return recall, ndcg, mrr
        
        
        
        
        

    


