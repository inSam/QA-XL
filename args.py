import argparse
import os

def setup_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train/Test a model on SQuAD')
    
    ### parameters ###
    parser.add_argument('--data_dir', default='./SQuAD/')
    parser.add_argument('--model_dir', default='train/best_model')
    parser.add_argument('--answer_file', default='test/' + 'train/best_model'.split('/')[-1] + '.answers')
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--epochs', type = int, default=20)
    parser.add_argument('--eval', type = bool, default=False)
    parser.add_argument('--load_model', type = bool, default = False)
    parser.add_argument('--batch_size', type = int, default=12)
    parser.add_argument('--grad_clipping', type = float, default = 10)
    parser.add_argument('--lrate', type = float, default=0.002)
    parser.add_argument('--dropout', type = float, default=0.3)
    parser.add_argument('--use_xl', type = bool, default=True)
    parser.add_argument('--fix_embeddings', type = bool, default=False)
    parser.add_argument('--char_dim', type = int, default=64)
    parser.add_argument('--pos_dim', type = int, default=12)
    parser.add_argument('--ner_dim', type = int, default=8)
    parser.add_argument('--char_hidden_size', type = int, default=50)
    parser.add_argument('--hidden_size', type = int, default=128)
    parser.add_argument('--attention_size', type = int, default=250)
    parser.add_argument('--decay_period', type = int, default=10)
    parser.add_argument('--decay', type = int, default=0.5)
    args = parser.parse_args()

    if not os.path.exists('train/'): ##train_model/
        os.makedirs('train/')
    if not os.path.exists('test/'): ##result/
        os.makedirs('test/')

    return args
