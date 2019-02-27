"""
Assortment of layer implementations specified in QANet
Reference: https://arxiv.org/pdf/1804.09541.pdf

Author: 
   Sam Xu (samx@stanford.edu)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from util import masked_softmax

class HighwayEncoder(nn.Module):
    """
    Edits: An dropout layer with p=0.1 was added

    Encode an input sequence using a highway network.
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
            # Shapes of g, t, and x are all (batch_size, hidden_size, seq_len)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            t = F.dropout(t, p=0.1, training=self.training)
            x = g * t + (1 - g) * x

        return x

    
class Initialized_Conv1d(nn.Module):
    """
    Wrapper Function
    Initializes nn.conv1d and adds a relu output.
    """
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

    Question: Linear/Conv1d layer before or after highway?
    """

    def __init__(self, p1, p2, hidden_size, dropout_w = 0.1, dropout_c = 0.05):
        super(Embedding, self).__init__()
        self.conv2d = nn.Conv2d(p2, hidden_size, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(p1 + hidden_size, hidden_size, bias=False)
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
        emb = self.conv1d(emb).transpose(1,2)
        #Emb: shape [batch_size * seq_len * hidden_size]
        print(emb.size())
        emb = self.high(emb).transpose(1,2)
        return emb

class DepthwiseSeperableConv(nn.Module):
    """
    Performs a depthwise seperable convolution
    First you should only convolve over each input channel individually, afterwards you convolve the input channels via inx1x1 to get the number of output channels
    This method conserves memory
    
    For clarification see the following: 
    https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
    https://arxiv.org/abs/1706.03059


    Args:
         in_channel (int): input channel
         out_channel (int): output channel
         k (int): kernel size

    Question: Padding in depthwise_convolution
    """
    def __init__(self, in_channel, out_channel, k, bias=True):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size = k, groups = in_channel, padding = k//2, bias = False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size = 1, bias=bias)

    def forward(self, input):
        return F.relu(self.pointwise_conv(self.depthwise_conv(input)))


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


class SelfAttention(nn.Module):
    """
    Implements the self-attention mechanism used in QANet. 

    Using the same implementation in "Attention" is all you need, we set value_dim = key_dim = d_model / num_head

    See references here: 
    https://arxiv.org/pdf/1706.03762.pdf
    https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#multi-head-self-attention

    Question: Do I use bias in the linear layers? 
    """
    def __init__(self, d_model, num_head, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.num_head = num_head
        self.kv_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

    def forward(self, x, mask):
        kv = self.kv_conv(x)
        query = self.query_conv(x)
        kv = kv.transpose(1,2)
        query = query.transpose(1,2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(kv, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, mask):
        logits = torch.matmul(q, k.permute(0,1,3,2))
        shapes = [x  if x != None else -1 for x in list(logits.size())]
        mask = mask.view(shapes[0], 1, 1, shapes[-1])
        logits = mask_logits(logits, mask)
        
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)        
        

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret
        

def PositionEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    Returns the position relative to a sinusoidal wave at varying frequency
    """
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal).transpose(1, 2)


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal
    
class Encoder(nn.Module):
    """
    Encoder structure specified in the QANet implementation

    Args:
         num_conv (int): number of depthwise convlutional layers
         d_model (int): size of model embedding
         num_head (int): number of attention-heads
         k (int): kernel size for convolutional layers
         dropout (float): layer dropout probability
    """

    def __init__(self, num_conv, d_model, num_head, k, dropout = 0.1):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([DepthwiseSeperableConv(d_model, d_model, k) for _ in range(num_conv)])
        self.conv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv)])

        self.att = SelfAttention(d_model, num_head, dropout = dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.num_conv = num_conv
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        """
        dropout probability: uses stochastic depth survival probability = 1 - (l/L)*pL, 
        reference here: https://arxiv.org/pdf/1603.09382.pdf 
        Question: uhhh you drop the whole layer apparently, and you apply dropout twice for each other layer?
        """
        total_layers = (self.num_conv + 1) * blks
        out = PositionEncoder(x)
        dropout = self.dropout

        for i, conv in enumerate(self.convs):
            res = out
            out = self.conv_norms[i](out.transpose(1,2)).transpose(1,2)
            if(i) % 2 == 0:
                out = F.dropout(out, p=dropout)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.res_drop(out, res, dropout*float(l)/total_layers)
            l += 1

        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.att(out, mask)
        out = self.res_drop(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.res_drop(out, res, dropout*float(l)/total_layers)
        return out

        
    def res_drop(self, x, res, drop):
        if self.training == True:
           if torch.empty(1).uniform_(0,1) < drop:
               return res
           else:
               return F.dropout(x, drop, training=self.training) + res
        else:
            return x + res

class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res        

class QAOutput(nn.Module):
    def __init__(self, hidden_size):
        super(QAOutput, self).__init__()
        self.w1 = Initialized_Conv1d(hidden_size*2, 1)
        self.w2 = Initialized_Conv1d(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2
        
