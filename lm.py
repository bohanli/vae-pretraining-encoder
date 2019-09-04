import os
import sys
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData

from modules import LSTM_LM

from exp_utils import create_exp_dir
from utils import uniform_initializer, xavier_normal_initializer

# clip_grad = 5.0
# decay_epoch = 2
# lr_decay = 0.5
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    parser.add_argument('--exp_dir', default=None, type=str,
                         help='experiment directory.')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # optimization parameters
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')

    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--decay_epoch', type=int, default=2)
    parser.add_argument('--clip_grad', type=float, default=5.0, help='')

    # others
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')

    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)


    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "exp_{}_lm/{}_{}_{}".format(args.dataset,
            args.dataset, args.opt, args.lr)

    if len(args.load_path) <= 0 and args.eval:
        args.load_path = os.path.join(args.exp_dir, 'model.pt')

    args.save_path = os.path.join(args.exp_dir, 'model.pt')

    return args

def test(model, test_data_batch, args):
    global logging

    report_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size


        loss = model.reconstruct_error(batch_data)


        loss = loss.sum()

        report_loss += loss.item()

    nll = (report_loss) / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    logging('avg_loss: %.4f, nll: %.4f, ppl: %.4f' % \
           (nll, nll, ppl))
    sys.stdout.flush()

    return nll, ppl

def main(args):
    global logging
    logging = create_exp_dir(args.exp_dir, scripts_to_save=["text_cyc_anneal.py"])

    if args.cuda:
        logging('using cuda')

    logging('model saving path: %s' % args.save_path)

    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": args.lr, "best_loss": 1e4}

    train_data = MonoTextData(args.train_data)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, vocab=vocab)
    test_data = MonoTextData(args.test_data, vocab=vocab)

    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    lm = LSTM_LM(args, vocab, model_init, emb_init).to(device)

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path)
        lm.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)

    if args.opt == "sgd":
        optimizer = optim.SGD(lm.parameters(), lr=args.lr, momentum=args.momentum)
        opt_dict['lr'] = args.lr
    elif args.opt == "adam":
        optimizer = optim.Adam(lm.parameters(), lr=args.lr)
        opt_dict['lr'] = args.lr
    else:
        raise ValueError("optimizer not supported")

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_nll = best_ppl = 0
    lm.train()
    start = time.time()

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)
    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)
    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)
    for epoch in range(args.epochs):
        report_loss = 0
        report_num_words = report_num_sents = 0
        for i in np.random.permutation(len(train_data_batch)):
            batch_data = train_data_batch[i]
            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size

            report_num_sents += batch_size

            optimizer.zero_grad()

            loss = lm.reconstruct_error(batch_data)

            report_loss += loss.sum().item()
            loss = loss.mean(dim=-1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), args.clip_grad)

            optimizer.step()

            if iter_ % args.log_niter == 0:
                train_loss = report_loss / report_num_sents

                logging('epoch: %d, iter: %d, avg_loss: %.4f, time elapsed %.2fs' %
                       (epoch, iter_, train_loss, time.time() - start))
                sys.stdout.flush()

            iter_ += 1

        if epoch % args.test_nepoch == 0:
            #logging('epoch: %d, testing' % epoch)
            lm.eval()

            with torch.no_grad():
                nll, ppl = test(lm, test_data_batch, args)
                logging('test | epoch: %d, nll: %.4f, ppl: %.4f' % (epoch, nll, ppl))
            lm.train()


        lm.eval()
        with torch.no_grad():
            nll, ppl = test(lm, val_data_batch, args)
            logging('valid | epoch: %d, nll: %.4f, ppl: %.4f' % (epoch, nll, ppl))

        if nll < best_loss:
            logging('update best loss')
            best_loss = nll
            best_nll = nll
            best_ppl = ppl
            torch.save(lm.state_dict(), args.save_path)

        if nll > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= args.decay_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * args.lr_decay
                lm.load_state_dict(torch.load(args.save_path))
                logging('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                if args.opt == "sgd":
                    optimizer = optim.SGD(lm.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                elif args.opt == "adam":
                    optimizer = optim.Adam(lm.parameters(), lr=opt_dict["lr"])
                else:
                    raise ValueError("optimizer not supported")
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = nll

        if decay_cnt == max_decay:
            break

        lm.train()

    logging('valid | best_loss: %.4f, nll: %.4f, ppl: %.4f' \
          % (best_loss, best_nll, best_ppl))
    
    # reload best lm model
    lm.load_state_dict(torch.load(args.save_path))
    
    with torch.no_grad():
        nll, ppl = test(lm, test_data_batch, args)
        logging('test | nll: %.4f, ppl: %.4f' % (nll, ppl))



if __name__ == '__main__':
    args = init_config()
    main(args)
