import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear

import math
from torch.autograd import Variable

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
        self.conv1d = Conv1d_Wrap(p1 + hidden_size, p1, bias=False)
        self.high = HighwayEncoder(2, p1)
        
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c


    def forward(self, ch, wd):
        # switch to [bsz x ch_size x seq_len x word_len]
        ch = ch.permute(0, 3, 1, 2)
        ch = F.dropout(ch, p=self.dropout_c, training=self.training)
        ch = self.conv2d(ch)
        ch = F.relu(ch)
        ch, _ = torch.max(ch, dim=3)

        #Feedforward with linear layer
        wd = F.dropout(wd, p=self.dropout_w, training=self.training)
        wd = wd.transpose(1, 2)
        emb = torch.cat([ch, wd], dim=1)
        emb = self.conv1d(emb).transpose(1,2)
        
        #Emb: shape [batch_size * seq_len * hidden_size]
        #print(emb.size())
        
        #highway embedding 
        emb = self.high(emb)
        return emb

class Conv1d_Wrap(nn.Module):
    """
    Wrapper Function
    Initializes nn.conv1d and adds a relu output.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super(Conv1d_Wrap, self).__init__()
        
        self.out = nn.Conv1d( in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
            self.relu = True
        else:
            nn.init.xavier_uniform_(self.out.weight)
            self.relu = False

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)
    

def Dropout(x, dropout, is_train, get_mask = False) :
    """
    Embedding dropout
    
    Randomly masks the embedding vector
    """

    if dropout > 0.0 and is_train :
        shape = x.size()
        placehold = Variable(torch.FloatTensor(shape[0], 1, shape[2]))
        placehold = placehold.cuda()
        nn.init.uniform_(placehold)
        random_tensor = (1.0 - dropout) + placehold
        mask_tensor = torch.floor(random_tensor)
        x = torch.div(x, 1.0 - dropout) * mask_tensor

    if get_mask:
        return mask_tensor
    
    return x


class StackedPaddedRNN(nn.Module):
    """
    A BiLSTM wrapper
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(StackedPaddedRNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(nn.LSTM(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask=None):
        if (x_mask is None) or (x_mask.data.sum() == 0) or (not self.training):
            return self._forward_unpadded(x, x_mask)
        
        return self._forward_padded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        outputs = [x.transpose(0, 1)]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout,
                                      training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        output = torch.cat(outputs[1:], 2)
        output = output.transpose(0, 1)
        return output

    def _forward_padded(self, x, x_mask):
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)

        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
                
            outputs.append(self.rnns[i](rnn_input)[0])


        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]


        output = torch.cat(outputs[1:], 2)
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        return output

class WordAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout) :
        super(WordAttention, self).__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, p, p_mask, q, q_mask, is_training):
        ## Applying embedding masking
        if is_training :
            drop_mask = Dropout(p, self.dropout, is_training, get_mask = True)
            d_p = torch.div(p, (1.0 - self.dropout)) * drop_mask
            d_ques = torch.div(q, (1.0 - self.dropout)) * drop_mask
        else :
            d_p = p
            d_ques = q

        Wp = F.relu(self.W(d_p))
        Wq = F.relu(self.W(d_ques))

        scores = torch.bmm(Wp, Wq.transpose(2, 1))
        mask = q_mask.unsqueeze(1).repeat(1, p.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        
        alpha = F.softmax(scores, dim=2)
        output = torch.bmm(alpha, q)

        return output

class MultiAttention(nn.Module) :

    def __init__(self, input_size, hidden_size, dropout):
        super(MultiAttention, self).__init__()
        self.dropout = dropout
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.U = nn.Linear(input_size, hidden_size, bias=False)
        self.D = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)

        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.U.weight.data)

    def forward(self, p, p_mask, q, q_mask, rep, rep_p, is_training, name = None):


        d_p = p
        d_q = q
        if is_training :
            drop_mask = Dropout(p, self.dropout, is_training, get_mask=True)
            d_p = torch.div(d_p, (1.0 - self.dropout)) * drop_mask
            d_q = torch.div(d_q, (1.0 - self.dropout)) * drop_mask


        Up = F.relu(self.U(d_p))
        Uq = F.relu(self.U(d_q))
        D = self.D.expand_as(Uq)

        Uq = D * Uq

        scores = Up.bmm(Uq.transpose(2, 1))

        if name is not None:
            print("score size: {}".format(scores.size()))
            your_file = open(name, 'ab')
            np.savetxt(your_file, scores.data.cpu()[0].numpy())
            your_file.close()

        output_q = None
        if rep_p is not None:
            scores_T = scores.clone()
            scores_T = scores_T.transpose(1, 2)
            mask_p = p_mask.unsqueeze(1).repeat(1, q.size(1), 1)
            scores_T.data.masked_fill_(mask_p.data, -float('inf'))
            alpha_p = F.softmax(scores_T, 2)
            output_q = torch.bmm(alpha_p, rep_p)

        mask = q_mask.unsqueeze(1).repeat(1, p.size(1), 1)
        scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = F.softmax(scores, 2)
        output = torch.bmm(alpha, rep)

        return output, output_q

class GetAnswer(nn.Module):
    """
    Bilinear attention layer over a sequence
    """

    def __init__(self, xdim, ydim, first=False):
        super(GetAnswer, self).__init__()
        self.linear = nn.Linear(ydim, xdim)
        self.rnn = nn.GRUCell(xdim, ydim)
        self.first = first
        
        if self.first:
            self.rnn = nn.GRUCell(xdim, ydim)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.1)

    def forward(self, x, y, x_mask):

        drop_y = F.dropout(y, p=self.opt['dropout'], training=self.training)
        yW = self.linear(drop_y)
        xWy = yW.unsqueeze(1).bmm(x.transpose(2, 1)).squeeze(1)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))

        if self.first:
            alpha = F.softmax(xWy, dim=1)

        y_new = None
        
        if self.first:
            rnn_input = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
            rnn_input = F.dropout(rnn_input, p=self.opt['dropout'], training=self.training)
            y_new = self.rnn(rnn_input, y)

        return xWy, y_new


class PointerNet(nn.Module) :
    """
    Calculates the logits for start and end boundary
    """

    def __init__(self, input_size, use_cuda = True) :
        super(PointerNet, self).__init__()
        self.input_size = input_size
        self.start = GetAnswer(input_size, input_size, first=True)
        self.end = GetAnswer(input_size, input_size,  first=False)

    def forward(self, self_states, p_mask, init_states, q_summ, is_training) :

        logits1, init_states = self.start(self_states, q_summ, p_mask)
        logits2, _ = self.end(self_states, q_summ, p_mask)
        return logits1, logits2


class PointerS(nn.Module) :
    """
    Calculates t from our final encoding
    """

    def __init__(self, input_size, dropout, use_cuda = True) :
        super(PointerS, self).__init__()
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.w = nn.Linear(input_size, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w.weight.data)
        self.w.bias.data.fill_(0.1)

    def forward(self, x, mask, is_training) :

        d_x = Dropout(x, self.dropout, is_training)
        beta = self.w(d_x).squeeze(2)
        beta.data.masked_fill_(mask.data, -float('inf'))
        beta = F.softmax(beta, 1)
        output = torch.bmm(beta.unsqueeze(1), x).squeeze(1)
        return output
