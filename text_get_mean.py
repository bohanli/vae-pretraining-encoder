import os, sys
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim

from data import MonoTextData
from modules import VAE
from modules import GaussianLSTMEncoder, LSTMDecoder

from exp_utils import create_exp_dir
from utils import uniform_initializer, xavier_normal_initializer, calc_iwnll, calc_mi, calc_au, sample_sentences, visualize_latent, reconstruct

# old parameters
clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5

# Junxian's new parameters
# clip_grad = 1.0
# decay_epoch = 5
# lr_decay = 0.5
# max_decay = 5

def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, required=True, help='dataset to use')
    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                         help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_dir', type=str, default='')

    # decoding
    parser.add_argument('--reconstruct_from', type=str, default='', help="the model checkpoint path")
    parser.add_argument('--reconstruct_to', type=str, default="decoding.txt", help="save file")
    parser.add_argument('--decoding_strategy', type=str, choices=["greedy", "beam", "sample"], default="greedy")

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10, help="number of annealing epochs. warm_up=0 means not anneal")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")


    # inference parameters
    parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')

    # output directory
    parser.add_argument('--exp_dir', default=None, type=str,
                         help='experiment directory.')
    parser.add_argument("--save_ckpt", type=int, default=0,
                        help="save checkpoint every epoch before this number")
    parser.add_argument("--save_latent", type=int, default=0)

    # new
    parser.add_argument("--fix_var", type=float, default=-1)
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--load_best_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1.)

    parser.add_argument("--fb", type=int, default=0,
                         help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
    parser.add_argument("--target_kl", type=float, default=-1,
                         help="target kl of the free bits trick")


    args = parser.parse_args()

    # set args.cuda
    args.cuda = torch.cuda.is_available()

    # set seeds
    # seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    # args.seed = seed_set[args.taskid]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    args.save_dir = args.load_dir
    args.load_path = os.path.join(args.load_dir, "model.pt")

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args


def test(model, test_data_batch, mode, args, verbose=True):
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        loss, loss_rc, loss_kl = model.loss(batch_data, 1.0, nsamples=args.nsamples)
        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        if args.warm_up == 0 and args.kl_start < 1e-6:
            report_loss += loss_rc.item()
        else:
            report_loss += loss.item()

    mutual_info = calc_mi(model, test_data_batch)

    test_loss = report_loss / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    if verbose:
        print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
               (mode, test_loss, report_kl_loss / report_num_sents, mutual_info,
                report_rec_loss / report_num_sents, nll, ppl))
        #sys.stdout.flush()

    return test_loss, nll, kl, ppl, mutual_info


def save_latents(args, vae, test_data_batch, test_label_batch, str_):
    fout_label = open(os.path.join(args.save_dir, f'{str_}.label'),'w')
    with open(os.path.join(args.save_dir, f'{str_}.vec'),'w') as f:
        for i in range(len(test_data_batch)):
            batch_data = test_data_batch[i]
            batch_label = test_label_batch[i]
            batch_size, sent_len = batch_data.size()
            means, _ = vae.encoder.forward(batch_data)
            for j in range(batch_size):
                fout_label.write(batch_label[j] + "\n")
                mean = means[j,:].cpu().detach().numpy().tolist()
                f.write('\t'.join([str(val) for val in mean]) + '\n')


def main(args):
    train_data = MonoTextData(args.train_data, label=args.label)
    vocab = train_data.vocab
    vocab_size = len(vocab)
    
    vocab_path = os.path.join("/".join(args.train_data.split("/")[:-1]), "vocab.txt")
    with open(vocab_path, "w") as fout:
        for i in range(vocab_size):
            fout.write("{}\n".format(vocab.id2word(i)))
        #return

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()

    log_niter = (len(train_data)//args.batch_size)//10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    args.device = device

    if args.enc_type == 'lstm':
        encoder = GaussianLSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)
    vae = VAE(encoder, decoder, args).to(device)

    print('begin evaluation')
    vae.load_state_dict(torch.load(args.load_path))
    vae.eval()
    with torch.no_grad():
        test_data_batch, test_batch_labels = test_data.create_data_batch_labels(batch_size=args.batch_size,
                                                      device=device,
                                                      batch_first=True)

        # test(vae, test_data_batch, "TEST", args)
        # au, au_var = calc_au(vae, test_data_batch)
        # print("%d active units" % au)

        train_data_batch, train_batch_labels = train_data.create_data_batch_labels(batch_size=args.batch_size,
                                                        device=device,
                                                        batch_first=True)

        val_data_batch, val_batch_labels = val_data.create_data_batch_labels(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

        print("getting  vectors for training")
        save_latents(args, vae, train_data_batch, train_batch_labels, "train")
        print("getting  vectors for validating")
        save_latents(args, vae, val_data_batch, val_batch_labels, "val")
        print("getting  vectors for testing")
        save_latents(args, vae, test_data_batch, test_batch_labels, "test")


if __name__ == '__main__':
    args = init_config()
    main(args)
