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
                 d_model, train_cemb = False, pad=0, dropout=0.1, num_head = 8):
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
        
        wemb_dim = word_vectors.size()[1]
        cemb_dim = char_vectors.size()[1]
        #print("Word vector dim-%d, Char vector dim-%d" % (wemb_dim, cemb_dim))
        self.emb = layers.Embedding(wemb_dim, cemb_dim, d_model)
        self.emb_enc = layers.Encoder(num_conv=4, d_model=d_model, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = layers.CQAttention(d_model=d_model)
        self.cq_resizer = layers.Initialized_Conv1d(d_model * 4, d_model) #Foward layer to reduce dimension of cq_att output back to d_dim
        self.model_enc_blks = nn.ModuleList([layers.Encoder(num_conv=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1) for _ in range(7)])

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
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0

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

    num_head = 8
    qanet = QANet(wv_tensor, cv_tensor,
                  c_max_len, q_max_len, d_model, train_cemb=False, num_head=num_head)
    qanet(context_wids, context_cids,
                   question_wids, question_cids)
    
