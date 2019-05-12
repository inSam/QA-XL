import argparse
import torch
import pickle as pkl
import sys
import ujson as json
from model import QAxl
from util import data_loader, get_batches
from args import setup_args

def train(args):        
    torch.manual_seed(args.seed)

    # Get data loader
    train_data, dev_data, word2id, id2word, char2id, new_args = data_loader(args)
    model = QAxl(new_args)
    
    if args.use_cuda :
        model = model.cuda()

    dev_batches = get_batches(dev_data, args.batch_size, evaluation=True)

    # Get optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adamax(parameters, lr = args.lrate)
    lrate = args.lrate
    
    if args.eval :
        model.load_state_dict(torch.load(args.model_dir))
        model.eval()
        model.SelfEvaluate(dev_batches, args.data_dir + 'dev_eval.json', answer_file = args.answer_file, drop_file=args.data_dir + 'drop.json', dev=args.data_dir + 'dev.json')
        exit()

    if args.load_model:
        model.load_state_dict(torch.load(args.model_dir))

    best_score = 0.0

    ## Training
    for epoch in range(1, args.epochs + 1) :
        train_batches = get_batches(train_data, args.batch_size)
        dev_batches = get_batches(dev_data, args.batch_size, evaluation=True)

        model.train()
        for i, train_batch in enumerate(train_batches):
            loss = model(train_batch)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, new_args['grad_clipping'])
            optimizer.step()
            model.reset_parameters()

            if i % 100 == 0:
                print('epoch = %d,  loss = %.5f, step = %d, lrate = %.5f best_score = %.3f' % (epoch, model.train_loss.value, i, lrate, best_score))
                sys.stdout.flush()

        model.eval()
        exact_match_score, F1 = model.SelfEvaluate(dev_batches, args.data_dir + 'dev_eval.json', answer_file = args.answer_file, drop_file=args.data_dir + 'drop.json', dev=args.data_dir + 'dev-v2.0.json')

        if best_score < F1:
            best_score = F1
            print('saving %s ...' % args.model_dir)
            torch.save(model.state_dict(), args.model_dir)
        if epoch > 0 and epoch % args.decay_period == 0:
            lrate *= args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrate


if __name__ == '__main__':
    train(setup_args())
