"""
Assortment of layer implementations specified in QANet
Reference: https://arxiv.org/pdf/1804.09541.pdf

Author: 
   Sam Xu (samx@stanford.edu)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from util import masked_softmax

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.
    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).
    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            t = F.dropout(t, p=0.1, training=self.training)
            x = g * t + (1 - g) * x

        return x

    
class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super(Initialized_Conv1d, self).__init__()
        
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class Embedding(nn.Module):
    """
    Embedding layer specified by QANet. 
    Concatenation of 300-dimensional (p1) pre-trained GloVe word vectors and 200-dimensional (p2) trainable char-vectors 
    Char-vectors have a set length of 16 via truncation or padding. Max value is taken by each row/char? 
    To obtain a vector of (p1 + p2) long word vector 
    Uses two-layer highway network (Srivastava 2015) 

    Note: Dropout was used on character_word embeddings and between layers, specified as 0.1 and 0.05 respectively
    """

    def __init__(self, p1, p2, hidden_size, dropout_w = 0.1, dropout_c = 0.05):
        super(Embedding, self).__init__()
        self.conv2d = nn.Conv2d(cemb_dim, hidden_size, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(wemb_dim + hidden_size, hidden_size, bias=False)
        self.high = HighwayEncoder(2, hidden_size)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c


    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb
        
