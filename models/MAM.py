import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import pdb

class MAM(torch.nn.Module):
    def __init__(self, item_num, device, args):
        super(MAM, self).__init__()

        self.item_num = item_num
        self.dim = args.dim
        self.model = args.model

        self.dev = device
        self.indices = torch.nn.Parameter(
            torch.arange(item_num+1, dtype=torch.long), requires_grad=False
        )

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.dim, padding_idx=0)
        self.key = torch.nn.Parameter(data=torch.rand((1, args.dim), dtype=torch.float), requires_grad=True)
        self.pos_emb = torch.nn.Embedding(args.max_len, args.dim)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.h = args.num_heads

        self.last_layernorm = torch.nn.LayerNorm(args.dim, eps=1e-8)
        self.attn_layernorm = torch.nn.LayerNorm(args.dim, eps=1e-8)
        self.attn_layer =  torch.nn.MultiheadAttention(args.dim, args.num_heads)
        self.fwd_layernorm = torch.nn.LayerNorm(args.dim, eps=1e-8)

    def dot_product_attention(self, q, k, v, mask=None, batch_first=True):
        
        if not batch_first:
            q, k, v = q.permute(1,0,2), k.permute(1,0,2), v.permute(1,0,2)

        sim = (q * k).sum(-1)
        if mask != None:
            sim = sim + (~mask * -1e15)
        attn = F.softmax(sim, dim=-1)
        Q = (v * attn.unsqueeze(-1)).sum(1)
        return Q, attn

    def log2feats(self, log_seqs, isEval):
        seqs = self.item_emb(log_seqs)

        mask = log_seqs != 0
        if self.model == 'oracle':
            mask[:,-1] = False

        seqs *= self.item_emb.embedding_dim ** 0.5

        if self.model != 'oracle' and self.model != 'mean':
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))

        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        if self.model == 'oracle':
            last = seqs[:,-1:,:]
            hn, attn = self.dot_product_attention(last, seqs, seqs, mask)
            Q = hn
        elif self.model == 'mean':
            hn = seqs.sum(1) / mask.sum(-1).unsqueeze(1)
            Q  = hn
            attn = 0.0
        else:
            hn, attn = self.dot_product_attention(self.key.unsqueeze(0), seqs, seqs, mask)
            Q = hn

        if self.model == 'P2MAMO' or self.model == 'oracle' or self.model == 'mean':
            return Q, hn, attn, 0.0

        seqs = torch.transpose(seqs, 0, 1)
        Q = self.attn_layernorm(Q).unsqueeze(1)
        Q = torch.transpose(Q, 0, 1)

        mha_outputs, attn_output_weights = self.attn_layer(Q, seqs, seqs, key_padding_mask=~mask)

        if self.model == 'P2MAMOP':
            Q = Q + mha_outputs
        elif self.model == 'P2MAMP':
            Q = mha_outputs

        Q = torch.transpose(Q, 0, 1)
        Q = self.fwd_layernorm(Q)
        log_feats = Q

        return log_feats, hn, attn, attn_output_weights.squeeze()

    def forward(self, log_seqs, pred, isEval):

        log_feats, hn, attn, attn_output_weights = self.log2feats(log_seqs, isEval)

        if not isEval:
            
            all_items_emb = self.item_emb(self.indices)
            pos_logits_prev = log_feats.squeeze() @ all_items_emb.t()
            pos_logits_sf = F.softmax(pos_logits_prev, dim=-1)
            pos_logits = torch.gather(pos_logits_sf, 1, pred)
            return pos_logits

        else:
            emb_pred_item = self.item_emb(self.indices).squeeze()
            log_feats = log_feats.squeeze()
            #rank will not change in softmax so we don't have softmax in testing
            logits = torch.matmul(log_feats, emb_pred_item.t())
             
            return logits
