import torch
import random

import argparse
import collections
import json
import ujson
import numpy as np
import os
import re
import string
import sys
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from pytorch_pretrained_bert import TransfoXLTokenizer
import torch.nn.functional as F


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.beta = 0.99
        self.moment = 0
        self.value = 0
        self.count = 0

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)

    def update(self, val):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """

        self.count += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        self.value = self.moment / (1 - self.beta ** self.count)

def read_json(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = json.load(f)
    return data

def get_data(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = ujson.load(f)
    return data


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor

cf_right = 0

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid_to_has_ans[qa['id']] = bool(qa['answers'])
  return qid_to_has_ans
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  global cf_right
  f1_scores = {}
  for article in dataset:
    for p in article['paragraphs']:
      for qa in p['qas']:
        qid = qa['id']
        gold_answers = [a['text'] for a in qa['answers']
                        if normalize_answer(a['text'])]
        if not gold_answers:
          # For unanswerable questions, only correct answer is empty string
          gold_answers = ['']
        if qid not in preds:
          print('Missing prediction for %s' % qid)
          continue
        a_pred = preds[qid]
        if (len(gold_answers) == len(a_pred) == 0) or (len(gold_answers) > 0 and len(a_pred) > 0):
            cf_right += 1
        # Take max over all gold answers
        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
  new_scores = {}
  for qid, s in scores.items():
    pred_na = na_probs[qid] > na_prob_thresh
    if pred_na:
      new_scores[qid] = float(not qid_to_has_ans[qid])
    else:
      new_scores[qid] = s
  return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  if not qid_list:
    total = len(exact_scores)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores.values()) / total),
        ('f1', 100.0 * sum(f1_scores.values()) / total),
        ('total', total),
    ])
  else:
    total = len(qid_list)
    return collections.OrderedDict([
        ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
        ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
        ('total', total),
    ])

def merge_eval(main_eval, new_eval, prefix):
  for k in new_eval:
    main_eval['%s_%s' % (prefix, k)] = new_eval[k]

def plot_pr_curve(precisions, recalls, out_image, title):
  plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
  plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.title(title)
  plt.savefig(out_image)
  plt.clf()

def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
                               out_image=None, title=None):
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  true_pos = 0.0
  cur_p = 1.0
  cur_r = 0.0
  precisions = [1.0]
  recalls = [0.0]
  avg_prec = 0.0
  for i, qid in enumerate(qid_list):
    if qid_to_has_ans[qid]:
      true_pos += scores[qid]
    cur_p = true_pos / float(i+1)
    cur_r = true_pos / float(num_true_pos)
    if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
      # i.e., if we can put a threshold after this point
      avg_prec += cur_p * (cur_r - recalls[-1])
      precisions.append(cur_p)
      recalls.append(cur_r)
  if out_image:
    plot_pr_curve(precisions, recalls, out_image, title)
  return {'ap': 100.0 * avg_prec}

def run_precision_recall_analysis(main_eval, exact_raw, f1_raw, na_probs, 
                                  qid_to_has_ans, out_image_dir):
  if out_image_dir and not os.path.exists(out_image_dir):
    os.makedirs(out_image_dir)
  num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
  if num_true_pos == 0:
    return
  pr_exact = make_precision_recall_eval(
      exact_raw, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_exact.png'),
      title='Precision-Recall curve for Exact Match score')
  pr_f1 = make_precision_recall_eval(
      f1_raw, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_f1.png'),
      title='Precision-Recall curve for F1 score')
  oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
  pr_oracle = make_precision_recall_eval(
      oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
      out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
      title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')
  merge_eval(main_eval, pr_exact, 'pr_exact')
  merge_eval(main_eval, pr_f1, 'pr_f1')
  merge_eval(main_eval, pr_oracle, 'pr_oracle')

def histogram_na_prob(na_probs, qid_list, image_dir, name):
  if not qid_list:
    return
  x = [na_probs[k] for k in qid_list]
  weights = np.ones_like(x) / float(len(x))
  plt.hist(x, weights=weights, bins=20, range=(0.0, 1.0))
  plt.xlabel('Model probability of no-answer')
  plt.ylabel('Proportion of dataset')
  plt.title('Histogram of no-answer probability: %s' % name)
  plt.savefig(os.path.join(image_dir, 'na_prob_hist_%s.png' % name))
  plt.clf()

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh

def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh


def evaluate(dataset_json, preds):
    dataset = dataset_json['data']
    na_probs = {k: 0.0 for k in preds}
    qid_to_has_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, 1.0)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, 1.0)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    return out_eval


def calc_mask(x):
    mask = torch.eq(x, 0)
    mask = mask.cuda()
    return mask


def prepare_data(batch_data):
    passage_ids = Variable(torch.LongTensor(batch_data[0]))
    passage_char_ids = Variable(torch.LongTensor(batch_data[1]))
    passage_pos_ids = Variable(torch.LongTensor(batch_data[2]))
    passage_ner_ids = Variable(torch.LongTensor(batch_data[3]))
    passage_match_origin = Variable(torch.LongTensor(batch_data[4]))
    passage_match_lower = Variable(torch.LongTensor(batch_data[5]))
    passage_match_lemma = Variable(torch.LongTensor(batch_data[6]))
    passage_tf = Variable(torch.FloatTensor(batch_data[7]))
    ques_ids = Variable(torch.LongTensor(batch_data[8]))
    ques_char_ids = Variable(torch.LongTensor(batch_data[9]))
    ques_pos_ids = Variable(torch.LongTensor(batch_data[10]))
    ques_ner_ids = Variable(torch.LongTensor(batch_data[11]))
    passage_xl_ids = pad_sequence(batch_data[15], batch_first=True)
    ques_xl_ids = pad_sequence(batch_data[16], batch_first=True)
    has_ans = Variable(torch.LongTensor(batch_data[17]))
    ques_tf = Variable(torch.FloatTensor(batch_data[18]))
    ques_match_origin = Variable(torch.LongTensor(batch_data[19]))
    ques_match_lower = Variable(torch.LongTensor(batch_data[20]))
    ques_match_lemma = Variable(torch.LongTensor(batch_data[21]))
    y1 = Variable(torch.LongTensor(batch_data[12]))
    y2 = Variable(torch.LongTensor(batch_data[13]))
    y1p = Variable(torch.LongTensor(batch_data[22]))
    y2p = Variable(torch.LongTensor(batch_data[23]))

    q_len = len(batch_data[21][0])
    p_lengths = passage_ids.ne(0).long().sum(1)
    q_lengths = ques_ids.ne(0).long().sum(1)
    passage_maxlen = int(torch.max(p_lengths, 0)[0])
    ques_maxlen = int(torch.max(q_lengths, 0)[0])
    #print(q_len)
    #print(ques_maxlen)

    passage_ids = passage_ids[:, :passage_maxlen]
    passage_char_ids = passage_char_ids[:, :passage_maxlen]
    passage_pos_ids = passage_pos_ids[:, :passage_maxlen]
    passage_ner_ids = passage_ner_ids[:, :passage_maxlen]
    passage_match_origin = passage_match_origin[:, :passage_maxlen]
    passage_match_lower = passage_match_lower[:, :passage_maxlen]
    passage_match_lemma = passage_match_lemma[:, :passage_maxlen]
    passage_tf = passage_tf[:, :passage_maxlen]

    #print(ques_xl_ids.shape())
    ques_xl_ids = ques_xl_ids[:, q_len - ques_maxlen:]
    ques_ids = ques_ids[:, q_len - ques_maxlen:]
    ques_char_ids = ques_char_ids[:, q_len - ques_maxlen:, :]
    ques_pos_ids = ques_pos_ids[:, q_len - ques_maxlen:]
    ques_ner_ids = ques_ner_ids[:, q_len - ques_maxlen:]
    ques_match_origin = ques_match_origin[:, q_len - ques_maxlen:]
    ques_match_lower = ques_match_lower[:, q_len - ques_maxlen:]
    ques_match_lemma = ques_match_lemma[:, q_len - ques_maxlen:]
    ques_tf = ques_tf[:, q_len - ques_maxlen:]

    p_mask = calc_mask(passage_ids)
    q_mask = calc_mask(ques_ids)
    cat_mask = torch.cat([torch.zeros_like(q_mask).byte(), p_mask], 1)
    m = torch.zeros(q_mask.size(0)).byte().view(-1, 1).cuda()
    q_mask = torch.cat([q_mask, m], 1)

    passage_ids = torch.cat([ques_ids, passage_ids], dim=1)
    passage_char_ids = torch.cat([ques_char_ids, passage_char_ids], dim=1)
    passage_pos_ids = torch.cat([ques_pos_ids, passage_pos_ids], dim=1)
    passage_ner_ids = torch.cat([ques_ner_ids, passage_ner_ids], dim=1)
    passage_match_origin = torch.cat([ques_match_origin, passage_match_origin], dim=1)
    passage_match_lower = torch.cat([ques_match_lower, passage_match_lower], dim=1)
    passage_match_lemma = torch.cat([ques_match_lemma, passage_match_lemma], dim=1)
    passage_tf = torch.cat([ques_tf, passage_tf], dim=1)


    passage_ids = passage_ids.cuda()
    passage_char_ids = passage_char_ids.cuda()
    passage_xl_ids = passage_xl_ids.cuda()
    ques_xl_ids = ques_xl_ids.cuda()
    passage_pos_ids = passage_pos_ids.cuda()
    passage_ner_ids = passage_ner_ids.cuda()
    passage_match_origin = passage_match_origin.cuda()
    passage_match_lower = passage_match_lower.cuda()
    passage_match_lemma = passage_match_lemma.cuda()
    passage_tf = passage_tf.cuda()
    y1 = y1.cuda()
    y2 = y2.cuda()
    y1p = y1p.cuda()
    y2p = y2p.cuda()
    has_ans = has_ans.cuda()

    batch_data = {
        "passage_ids": passage_ids,
        "passage_char_ids": passage_char_ids,
        "passage_xl_ids" : passage_xl_ids,
        "ques_xl_ids" : ques_xl_ids,
        "passage_pos_ids": passage_pos_ids,
        "passage_ner_ids": passage_ner_ids,
        "passage_match_origin": passage_match_origin.unsqueeze(2).float(),
        "passage_match_lower": passage_match_lower.unsqueeze(2).float(),
        "passage_match_lemma": passage_match_lemma.unsqueeze(2).float(),
        "passage_tf" : passage_tf.unsqueeze(2),
        "p_mask" : p_mask,
        "q_mask" : q_mask,
        "y1": y1,
        "y2": y2,
        "y1p": y1p,
        "y2p": y2p,
        "id" : batch_data[14],
        "has_ans": has_ans,
        "q_len": ques_maxlen,
        "cat_mask": cat_mask
    }

    return batch_data

def data_loader(args):
    print('loading data')
    opts = vars(args)
    
    data_dir = opts['data_dir']
    train_data = get_data(data_dir + 'train.json')
    dev_data = get_data(data_dir + 'dev.json')
    #train_data = get_data(data_dir + 'dev.json')
    #dev_data = get_data(data_dir + 'test.json')
    word2id = get_data(data_dir + 'word2id.json')
    id2word = {v : k for k, v in word2id.items()}
    char2id = get_data(data_dir + 'char2id.json')
    pos2id = get_data(data_dir + 'pos2id.json')
    ner2id = get_data(data_dir + 'ner2id.json')

    opts['char_size'] = int(np.max(list(char2id.values())) + 1)
    opts['pos_size'] = int(np.max(list(pos2id.values())) + 1)
    opts['ner_size'] = int(np.max(list(ner2id.values())) + 1)

    return train_data, dev_data, word2id, id2word, char2id, opts



def get_batches(data, batch_size, evaluation=False) :
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    
    if not evaluation:
        indices = list(range(len(data['context_ids'])))
        random.shuffle(indices)
        for key in data.keys():
            if isinstance(data[key], int):
                continue
            data[key] = [data[key][i] for i in indices]

    for i in range(0, len(data['context_ids']), batch_size) :
        batch_size = len(data['context_ids'][i:i+batch_size])
        yield (data['context_ids'][i:i+batch_size],
               data['context_char_ids'][i:i+batch_size],
               data['context_pos_ids'][i:i+batch_size],
               data['context_ner_ids'][i:i+batch_size],
               data['context_match_origin'][i:i+batch_size],
               data['context_match_lower'][i:i+batch_size],
               data['context_match_lemma'][i:i+batch_size],
               data['context_tf'][i:i+batch_size],
               data['ques_ids'][i:i+batch_size],
               data['ques_char_ids'][i:i+batch_size],
               data['ques_pos_ids'][i:i+batch_size],
               data['ques_ner_ids'][i:i+batch_size],
               data['y1'][i:i+batch_size],
               data['y2'][i:i+batch_size],
               data['id'][i:i+batch_size],
               #batch_to_ids(data['context_tokens'][i:i+batch_size]),
               #batch_to_ids(data['ques_tokens'][i:i+batch_size]),
               [torch.tensor(tokenizer.convert_tokens_to_ids(batch)) for batch in (data['context_tokens'][i:i+batch_size])],
               [torch.tensor(tokenizer.convert_tokens_to_ids(batch)) for batch in (data['ques_tokens'][i:i+batch_size])],
               data['has_ans'][i:i+batch_size],
               data['ques_tf'][i:i+batch_size],
               data['ques_match_origin'][i:i+batch_size],
               data['ques_match_lower'][i:i+batch_size],
               data['ques_match_lemma'][i:i+batch_size],
               data['y1p'][i:i+batch_size],
               data['y2p'][i:i+batch_size])

def unpack_data(data):
        cat_ids = data['passage_ids']
        passage_char_ids = data['passage_char_ids']
        passage_xl_ids = data['passage_xl_ids']
        ques_xl_ids = data['ques_xl_ids']
        cat_pos_ids = data['passage_pos_ids']
        cat_ner_ids = data['passage_ner_ids']
        cat_match_origin = data['passage_match_origin']
        cat_match_lower = data['passage_match_lower']
        cat_match_lemma = data['passage_match_lemma']
        cat_tf = data['passage_tf']
        p_mask = data['p_mask']
        q_len = data['q_len']
        q_mask = data['q_mask']
        cat_mask = data['cat_mask']
        return cat_ids, passage_char_ids, passage_xl_ids, ques_xl_ids, cat_pos_ids, cat_ner_ids, cat_match_origin, cat_match_lower, cat_match_lemma, cat_tf, p_mask, q_len, q_mask, cat_mask


def get_predictions( logits1, logits2, ans_log, maxlen=15) :

    batch_size, P = logits1.size()
    outer = torch.matmul(F.softmax(logits1, -1).unsqueeze(2),
                         F.softmax(logits2, -1).unsqueeze(1))

    vec_mask = Variable(torch.zeros(P, P))


    vec_mask = vec_mask.cuda()

    for j in range(P-1) :
        i = j + 1
        vec_mask[i, i:max(i+maxlen, P)].data.fill_(1.0)

    vec_mask = vec_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    outer = outer * vec_mask

    yp1 = torch.max(torch.max(outer, 2)[0], 1)[1]
    yp2 = torch.max(torch.max(outer, 1)[0], 1)[1]


    sm = F.softmax(ans_log, dim=-1)
    sm[:, 0] += 0.13
    sm[:, 1] -= 0.13
    has_ans = torch.max(sm, -1)[1]

    return yp1, yp2, has_ans

def convert_tokens(eval_file, qa_id, pp1, pp2, has_ans) :
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2, has in zip(qa_id, pp1, pp2, has_ans) :

        p1 = int(p1)
        p2 = int(p2)
        if not int(has):
            p1 = p2 = 0
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        # print("p1 value {}".format(p1))
        # print(spans)
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx : end_idx]
        remapped_dict[uuid] = context[start_idx : end_idx]
        if p1 == 0 and p2 == 0:
            answer_dict[str(qid)] = ""
            remapped_dict[uuid] = ""

    return answer_dict, remapped_dict

def compute_loss(logits1, logits2, ans_log, y1, y2, has_ans):
    """
    Compute cross-entropy loss of logits
    """
    loss1 = loss2 = loss_ans = 0
    loss1 = F.cross_entropy(logits1, y1)
    loss2 = F.cross_entropy(logits2, y2)

    loss_ans = F.cross_entropy(ans_log, has_ans)
    loss = loss1 + loss2 + loss_ans
    return loss
