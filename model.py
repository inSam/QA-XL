import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F
import ujson as json


from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert import TransfoXLModel
from util import AverageMeter, read_json, evaluate, prepare_data, unpack_data, get_predictions, compute_loss, convert_tokens
from layers import Embedding, Dropout, StackedPaddedRNN, WordAttention, MultiAttention, PointerS, PointerNet

from os.path import join
import csv

class QAxl(nn.Module):
    
    def __init__(self, args):
        super(QAxl, self).__init__()
        
        hidden_size = args['hidden_size']
        dropout = args['dropout']
        attention_size = args['attention_size']
        word_emb = np.array(read_json(args['data_dir'] + 'word_emb.json'), dtype=np.float32)
        word_size = word_emb.shape[0]
        word_dim = word_emb.shape[1]
        char_dim = args['char_dim']
        char_len = len(read_json(args['data_dir'] + 'char2id.json'))
        pos_dim = args['pos_dim']
        ner_dim = args['ner_dim']

        self.args = args
        self.train_loss = AverageMeter()
        self.use_cuda = args['use_cuda']
        self.use_xl = args['use_xl']

        if self.use_xl:
            self.xl = TransfoXLModel.from_pretrained('transfo-xl-wt103')
            xl_dim = 1024
        
        ## Embedding Layer
        print('Building embedding...')
        self.word_embeddings = nn.Embedding(word_emb.shape[0], word_dim, padding_idx=0)
        self.word_embeddings.weight.data = torch.from_numpy(word_emb)
        self.char_embeddings = nn.Embedding(char_len, char_dim, padding_idx = 0)
        self.pos_embeddings = nn.Embedding(args['pos_size'], args['pos_dim'], padding_idx=0)
        self.ner_embeddings = nn.Embedding(args['ner_size'], args['ner_dim'], padding_idx=0)
        with open(args['data_dir'] + 'tune_word_idx.pkl', 'rb') as f :
            tune_idx = pkl.load(f)
        self.fixed_idx = list(set([i for i in range(word_size)]) - set(tune_idx))
        fixed_embedding = torch.from_numpy(word_emb)[self.fixed_idx]
        self.register_buffer('fixed_embedding', fixed_embedding)
        self.fixed_embedding = fixed_embedding
        
        low_p_dim = word_dim + word_dim + args['pos_dim'] + args['ner_dim'] + 4
        low_q_dim = word_dim + args['pos_dim'] + args['ner_dim']
        if self.use_xl :
            low_p_dim += xl_dim
            low_q_dim += xl_dim

        self.emb_char = Embedding(word_dim, char_dim, hidden_size)

        ## Forward Layers Declaration
        high_p_dim = 2 * hidden_size
        full_q_dim = 2 * high_p_dim
        attention_dim = word_dim + full_q_dim
        if self.use_xl :
            attention_dim += xl_dim
        
        self.word_attention_layer = WordAttention(word_dim, attention_size, dropout)

        self.low_rnn = StackedPaddedRNN(low_p_dim, hidden_size, 1, dropout=dropout)
        self.high_rnn = StackedPaddedRNN(high_p_dim, hidden_size, 1, dropout=dropout)
        self.full_rnn = StackedPaddedRNN(full_q_dim, hidden_size, 1, dropout=dropout)
        
        self.low_attention_layer = MultiAttention(attention_dim, attention_size, dropout)
        self.high_attention_layer = MultiAttention(attention_dim, attention_size, dropout)
        self.full_attention_layer = MultiAttention(attention_dim, attention_size, dropout)

        ## Fusion Layer and Final Attention + Final RNN
        fuse_dim = 10 * hidden_size
        self_attention_dim = 12 * hidden_size + word_dim + ner_dim + pos_dim +  1
        if self.use_xl :
            self_attention_dim += xl_dim

        self.fuse_rnn = StackedPaddedRNN(fuse_dim, hidden_size, 1, dropout = dropout)
        self.self_attention_layer = MultiAttention(self_attention_dim, attention_size, dropout)
        self.self_rnn = StackedPaddedRNN(4 *  hidden_size, hidden_size, 1, dropout = dropout)

        ## Verifier and output
        self.summ_layer = PointerS(2 * hidden_size, dropout=dropout, use_cuda=self.use_cuda)
        self.summ_layer2 = PointerS(2 * hidden_size, dropout=dropout, use_cuda=self.use_cuda)
        self.pointer_layer = PointerNet(2 * hidden_size, use_cuda=self.use_cuda)
        self.has_ans = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(6*hidden_size, 2))

        
    def reset_parameters(self) :
        self.word_embeddings.weight.data[self.fixed_idx] = self.fixed_embedding

    def encode(self, data):
        args = self.args
        dropout = args['dropout']
        ids, p_char_ids, p_xl_ids, q_xl_ids, pos_ids, ner_ids, match_origin, match_lower, match_lemma, tf, p_mask, q_len, q_mask, mask = unpack_data(data)
        
        ### Transformer-XL
        if self.use_xl :
            with torch.no_grad():
                q_xl, mems_1 = self.xl(q_xl_ids)
                p_xl, _ = self.xl(p_xl_ids, mems=mems_1)

            ### dropout
            q_xl = Dropout(q_xl, dropout, self.training)
            p_xl = Dropout(p_xl, dropout, self.training)
            xl = torch.cat([q_xl, p_xl], 1)

        ### obtain embeddings
        #concatenated emb size: [bsz x len x model_size]
        emb = self.word_embeddings(ids)
        char_emb = self.char_embeddings(p_char_ids)
        emb = self.emb_char(char_emb, emb)
        pos_emb = self.pos_embeddings(pos_ids)
        ner_emb = self.ner_embeddings(ner_ids)
        emb = Dropout(emb, dropout, self.training) #dropout

        ## Break down into passage and question
        p_emb = emb[:, q_len:]
        q_emb = emb[:, :q_len]

        p_ner_emb = ner_emb[:, q_len:]
        q_ner_emb = ner_emb[:, :q_len]

        p_pos_emb = pos_emb[:, q_len:]
        q_pos_emb = pos_emb[:, :q_len]

        p_tf = tf[:, q_len:]
        q_tf = tf[:, :q_len]

        p_match_origin = match_origin[:, q_len:]
        q_match_origin = match_origin[:, :q_len]

        p_match_lemma = match_lemma[:, q_len:]
        q_match_lemma = match_lemma[:, :q_len]

        p_match_lower = match_lower[:, q_len:]
        q_match_lower = match_lower[:, :q_len]

        ### Attention
        word_attention_outputs = self.word_attention_layer(p_emb, p_mask, emb[:, :q_len+1], q_mask, self.training)
        q_word_attention_outputs = self.word_attention_layer(emb[:, :q_len+1], q_mask, p_emb, p_mask, self.training)
        word_attention_outputs[:, 0] += q_word_attention_outputs[:, -1]
        q_word_attention_outputs = q_word_attention_outputs[:, :-1]

        p_word_inp = torch.cat([p_emb, p_pos_emb, p_ner_emb, word_attention_outputs, p_match_origin, p_match_lower, p_match_lemma, p_tf], dim=2)
        
        q_word_inp = torch.cat([q_emb, q_pos_emb, q_ner_emb, q_word_attention_outputs, q_match_origin, q_match_lower, q_match_lemma, q_tf], dim=2)

        if self.use_xl :
            p_word_inp = torch.cat([p_word_inp, p_xl], dim=2)
            q_word_inp = torch.cat([q_word_inp, q_xl], dim=2)

        ### Encoding into low, high and full
        word_inp = torch.cat([q_word_inp, p_word_inp], 1)
        low_states = self.low_rnn(word_inp, mask)

        high_states = self.high_rnn(low_states, mask)
        full_inp = torch.cat([low_states, high_states], dim=2)
        full_states = self.full_rnn(full_inp, mask)

        ### Attention
        HoW = torch.cat([emb, low_states, high_states], dim=2)
        if self.use_xl :
            HoW = torch.cat([HoW, xl], dim=2)

        p_HoW = HoW[:, q_len:]
        low_p_states = low_states[:, q_len:]
        high_p_states = high_states[:, q_len:]
        full_p_states = full_states[:, q_len:]

        q_HoW = HoW[:, :q_len+1]
        low_q_states = low_states[:, :q_len+1]
        high_q_states = high_states[:, :q_len+1]
        full_q_states = full_states[:, :q_len+1]

        low_attention_outputs, low_attention_q = self.low_attention_layer(p_HoW, p_mask, q_HoW, q_mask, low_q_states, low_p_states, self.training)
        high_attention_outputs, high_attention_q = self.high_attention_layer(p_HoW, p_mask, q_HoW, q_mask, high_q_states, high_p_states, self.training)
        full_attention_outputs, full_attention_q = self.full_attention_layer(p_HoW, p_mask, q_HoW, q_mask, full_q_states, full_p_states, self.training)
        
        low_attention_outputs[:, 0] += low_attention_q[:, -1]
        high_attention_outputs[:, 0] += high_attention_q[:, -1]
        full_attention_outputs[:, 0] += full_attention_q[:, -1]

        fuse_inp = torch.cat([low_p_states, high_p_states, low_attention_outputs, high_attention_outputs, full_attention_outputs], dim = 2)
        
        fuse_q = torch.cat([low_q_states, high_q_states, low_attention_q, high_attention_q, full_attention_q], dim = 2)

        fuse_inp[:,0] += fuse_q[:,-1]
        fuse_concat = torch.cat([fuse_q[:,:-1], fuse_inp], 1)
        fused_states = self.fuse_rnn(fuse_concat, mask)

        ### Self Attention

        HoW = torch.cat([emb, pos_emb, ner_emb, tf, fuse_concat, fused_states], dim=2)

        if self.use_xl :
            HoW = torch.cat([HoW, xl], dim=2)

        self_attention_outputs, _ = self.self_attention_layer(HoW, mask, HoW, mask, fused_states, None, self.training)
        self_inp = torch.cat([fused_states, self_attention_outputs], dim=2)
        full_states = self.self_rnn(self_inp, mask)
        full_p_states = full_states[:, q_len:]
        full_q_states = full_states[:, :q_len]
        
        return full_p_states, p_mask, full_q_states, q_mask

    def decode(self, full_p_states, p_mask, full_q_states, q_mask) :
        q_summ = self.summ_layer(full_q_states, q_mask[:, :-1], self.training)
        logits1, logits2 = self.pointer_layer.forward(full_p_states, p_mask, None, q_summ, self.training)

        alpha1 = F.softmax(logits1, dim=1)
        alpha2 = F.softmax(logits2, dim=1)

        p_avg1 = torch.bmm(alpha1.unsqueeze(1), full_p_states).squeeze(1)
        p_avg2 = torch.bmm(alpha2.unsqueeze(1), full_p_states).squeeze(1)
        p_avg = p_avg1 + p_avg2
        
        q_summ = self.summ_layer2(full_q_states, q_mask[:, :-1], self.training)
        
        first_word = full_p_states[:, 0, :]

        has_inp = torch.cat([p_avg, first_word, q_summ], -1)
        has_log = self.has_ans(has_inp)
        return logits1, logits2, has_log

    def forward(self, data):
        data = prepare_data(data)
        full_p_states, p_mask, full_q_states, q_mask = self.encode(data)
        logits1, logits2, has_log = self.decode(full_p_states, p_mask, full_q_states, q_mask)
        loss = compute_loss(logits1, logits2, has_log, data['y1'], data['y2'], data['has_ans'])
        
        self.train_loss.update(loss.data.item())
        del full_p_states, p_mask, full_q_states, q_mask, logits1, logits2, has_log
        return loss


    def SelfEvaluate(self, batches, eval_file=None, answer_file = None, drop_file=None, dev=None) :
        print('Starting evaluation')

        with open(eval_file, 'r', encoding='utf-8') as f :
            eval_file = json.load(f)
        with open(dev, 'r', encoding='utf-8') as f :
            dev = json.load(f)

        answer_dict = {}
        mapped_dict = {}

        for batch in batches :
            data = prepare_data(batch)
            full_p_states, p_mask, full_q_states, q_mask = self.encode(data)
            logits1, logits2, ans_log = self.decode(full_p_states, p_mask, full_q_states, q_mask)
            y1, y2, has_ans = get_predictions(logits1, logits2, ans_log)
            qa_id = data['id']
            answer_dict_, mapped_dict_ = convert_tokens(eval_file, qa_id, y1, y2, has_ans)
            answer_dict.update(answer_dict_)
            mapped_dict.update(mapped_dict_)

            
            del full_p_states, p_mask, full_q_states, q_mask, y1, y2, answer_dict_, mapped_dict_, has_ans, ans_log, logits1, logits2

        with open(drop_file, 'r', encoding='utf-8') as f :
            drop = json.load(f)
        for i in drop['drop_ids']:
            uuid = eval_file[str(i)]["uuid"]
            answer_dict[str(i)] = ''
            mapped_dict[uuid] = ''

        with open(answer_file, 'w', encoding='utf-8') as f:
            json.dump(mapped_dict, f)
        metrics = evaluate(dev, mapped_dict)

        # sub_path = join('./result/', "submit.csv")
        # #log.info('Writing submission file to {}...'.format(sub_path))
        # with open(sub_path, 'w') as csv_fh:
        #     csv_writer = csv.writer(csv_fh, delimiter=',')
        #     csv_writer.writerow(['Id', 'Predicted'])
        #     for uuid in sorted(mapped_dict):
        #         csv_writer.writerow([uuid, mapped_dict[uuid]])
        
        print("EM: {}, F1: {}, Has answer: {}, No answer: {}".format(
            metrics['exact'], metrics['f1'], metrics['HasAns_f1'], metrics['NoAns_f1']))

        return metrics['exact'], metrics['f1']
