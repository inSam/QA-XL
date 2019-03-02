"""
Top level module for the original QANet implementation

Author: 
   Sam Xu (samx@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

import layers

class QANet(nn.Module):
    """
    Original QANet top module using the specification form the original paper

    Follows the following high-levels of implementation:
    1. Input Embedding Layer
    2. Embedding Encoder Layer
    3. Context-Query Attention Layer
    4. Model Encoder Layer
    5. Output Layer

    Args: 
        word_vectors (torch.Tensor): pre-trained word vector
    """

    def __init__(self, word_vectors, char_vectors, context_max_len, query_max_len,
                 d_model, d_head, mem_len = 0, train_cemb = False, pad=0, dropout=0.1, num_head = 8):
        """
        """
        super(QANet, self).__init__()
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(char_vectors, freeze=False)
            print("Training char_embeddings")
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_vectors)

        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.LC = context_max_len
        self.LQ = query_max_len
        self.num_head = num_head
        self.pad = pad
        self.dropout = dropout
        self.mem_len = mem_len
        self.d_head = d_head
        
        wemb_dim = word_vectors.size()[1]
        cemb_dim = char_vectors.size()[1]
        #print("Word vector dim-%d, Char vector dim-%d" % (wemb_dim, cemb_dim))

        #Layer Declarations
        self.emb = layers.Embedding(wemb_dim, cemb_dim, d_model)
        self.emb_enc = layers.Encoder(4, num_head, d_model, d_head, d_inner = d_model * 4, k=7, dropout=0.1) #Hard coded
        self.cq_att = layers.CQAttention(d_model=d_model)
        self.cq_resizer = layers.Initialized_Conv1d(d_model * 4, d_model) #Foward layer to reduce dimension of cq_att output back to d_dim
        self.model_enc_blks = nn.ModuleList([layers.Encoder(2, num_head, d_model, d_head, d_inner = d_model * 4, k=5, dropout=0.1) for _ in range(7)])
        self.out = layers.QAOutput(d_model)

        self._create_parameters()

    def _create_parameters(self):
            self.pos_emb = layers.PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))        

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems


    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

        hids = []

        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems
    
        
    def forward(self, Cword, Cchar, Qword, Qchar):
        """
        """
        maskC = (torch.ones_like(Cword) *
                 self.pad != Cword).float()
        maskQ = (torch.ones_like(Qword) *
                 self.pad != Qword).float()
        
        Cw, Cc = self.word_emb(Cword), self.char_emb(Cchar)
        Qw, Qc = self.word_emb(Qword), self.char_emb(Qchar)
        C, Q = self.emb(Cc, Cw, self.LC), self.emb(Qc, Qw, self.LQ)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        # X = self.cq_att(Ce, Qe, maskC, maskQ)
        # M0 = self.cq_resizer(X)
        # M0 = F.dropout(M0, p=self.dropout, training=self.training)
        
        # for i, blk in enumerate(self.model_enc_blks):
        #      M0 = blk(M0, maskC, i*(2+2)+1, 7)
        # M1 = M0
        # for i, blk in enumerate(self.model_enc_blks):
        #      M0 = blk(M0, maskC, i*(2+2)+1, 7)
        # M2 = M0
        # M0 = F.dropout(M0, p=self.dropout, training=self.training)
        # for i, blk in enumerate(self.model_enc_blks):
        #      M0 = blk(M0, maskC, i*(2+2)+1, 7)
        # M3 = M0
        # p1, p2 = self.out(M1, M2, M3, maskC)
        # return p1, p2        

if __name__ == "__main__":
    torch.manual_seed(12)
    # device and data sizes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wemb_vocab_size = 5000
    wemb_dim = 300
    cemb_vocab_size = 94
    cemb_dim = 64
    d_model = 96
    batch_size = 32
    q_max_len = 50
    c_max_len = 400
    char_dim = 16
    num_head = 8
    d_head = 12
    mem_len = 32
    
    # fake embedding
    wv_tensor = torch.rand(wemb_vocab_size, wemb_dim)
    cv_tensor = torch.rand(cemb_vocab_size, cemb_dim)
    
    # fake input
    question_lengths = torch.LongTensor(batch_size).random_(1, q_max_len)
    question_wids = torch.zeros(batch_size, q_max_len).long()
    question_cids = torch.zeros(batch_size, q_max_len, char_dim).long()
    context_lengths = torch.LongTensor(batch_size).random_(1, c_max_len)
    context_wids = torch.zeros(batch_size, c_max_len).long()
    context_cids = torch.zeros(batch_size, c_max_len, char_dim).long()

    for i in range(batch_size):
        question_wids[i, 0:question_lengths[i]] = \
            torch.LongTensor(1, question_lengths[i]).random_(
                1, wemb_vocab_size)
        question_cids[i, 0:question_lengths[i], :] = \
            torch.LongTensor(1, question_lengths[i], char_dim).random_(
                1, cemb_vocab_size)
        context_wids[i, 0:context_lengths[i]] = \
            torch.LongTensor(1, context_lengths[i]).random_(
                1, wemb_vocab_size)
        context_cids[i, 0:context_lengths[i], :] = \
            torch.LongTensor(1, context_lengths[i], char_dim).random_(
                1, cemb_vocab_size)

    qanet = QANet(wv_tensor, cv_tensor,
                  c_max_len, q_max_len, d_model, d_head, mem_len=mem_len, train_cemb=False, num_head=num_head)
    qanet(context_wids, context_cids,
                       question_wids, question_cids)    
    # p1, p2 = qanet(context_wids, context_cids,
    #                    question_wids, question_cids)
    # print(p1.shape)
    # print(p2.shape)    
