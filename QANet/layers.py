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
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
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
        emb = self.conv1d(emb)
        emb = self.high(emb)
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
        self.depthwise_conv = nn.conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size = k, groups = in_channels, padding = k//2, bias = False)
        self.pointwise_conv = nn.conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size = 1, bias=bias)

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
        super(selfAttention, self).__init__()
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
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, Q, K, V, mask):
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
        
