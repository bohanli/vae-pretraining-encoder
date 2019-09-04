import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim

from collections import defaultdict

from data import MonoTextData, VocabEntry
from modules import VAE, LinearDiscriminator, MLPDiscriminator
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

logging = None

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
    parser.add_argument('--load_path', type=str, default='')

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
    parser.add_argument("--load_best_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.)

    parser.add_argument("--fb", type=int, default=0,
                         help="0: no fb; 1: fb; 2: max(target_kl, kl) for each dimension")
    parser.add_argument("--target_kl", type=float, default=-1,
                         help="target kl of the free bits trick")

    parser.add_argument("--batch_size", type=int, default=16,
                         help="target kl of the free bits trick")    
    parser.add_argument("--update_every", type=int, default=1,
                         help="target kl of the free bits trick")  
    parser.add_argument("--num_label", type=int, default=100,
                         help="target kl of the free bits trick") 
    parser.add_argument("--freeze_enc", action="store_true", default=False)
    parser.add_argument("--discriminator", type=str, default="linear")

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
    if args.num_label == 100:
        params = importlib.import_module(config_file).params_ss_100
    elif args.num_label == 500:
        params = importlib.import_module(config_file).params_ss_500
    elif args.num_label == 1000:
        params = importlib.import_module(config_file).params_ss_1000
    elif args.num_label == 2000:
        params = importlib.import_module(config_file).params_ss_2000
    elif args.num_label == 10000:
        params = importlib.import_module(config_file).params_ss_10000

    args = argparse.Namespace(**vars(args), **params)

    load_str = "_load" if args.load_path != "" else ""
    if args.fb == 0:
        fb_str = ""
    elif args.fb == 1:
        fb_str = "_fb"
    elif args.fb == 2:
        fb_str = "_fbdim"

    opt_str = "_adam" if args.opt == "adam" else "_sgd"
    nlabel_str = "_nlabel{}".format(args.num_label)
    freeze_str = "_freeze" if args.freeze_enc else ""

    if len(args.load_path.split("/")) > 2:
        load_path_str = args.load_path.split("/")[1]
    else:
        load_path_str = args.load_path.split("/")[0]

    model_str = "_{}".format(args.discriminator)
    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "exp_{}{}_ss_ft/{}{}{}{}{}".format(args.dataset,
            load_str, load_path_str, model_str, opt_str, nlabel_str, freeze_str)


    if len(args.load_path) <= 0 and args.eval:
        args.load_path = os.path.join(args.exp_dir, 'model.pt')

    args.save_path = os.path.join(args.exp_dir, 'model.pt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    return args


def test(model, test_data_batch, test_labels_batch, mode, args, verbose=True):
    global logging

    report_correct = report_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_labels = test_labels_batch[i]
        batch_labels = [int(x) for x in batch_labels]

        batch_labels = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=args.device)

        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        loss, correct = model.get_performance(batch_data, batch_labels)

        loss = loss.sum()

        report_loss += loss.item()
        report_correct += correct

    test_loss = report_loss / report_num_sents
    acc = report_correct / report_num_sents

    if verbose:
        logging('%s --- avg_loss: %.4f, acc: %.4f' % \
               (mode, test_loss, acc))
        #sys.stdout.flush()

    return test_loss, acc


def main(args):
    global logging
    logging = create_exp_dir(args.exp_dir, scripts_to_save=[])

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    vocab = {}
    with open(args.vocab_file) as fvocab:
        for i, line in enumerate(fvocab):
            vocab[line.strip()] = i

    vocab = VocabEntry(vocab)

    train_data = MonoTextData(args.train_data, label=args.label, vocab=vocab)

    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)
    #sys.stdout.flush()

    log_niter = max(1, (len(train_data)//(args.batch_size * args.update_every))//10)

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

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path)
        #curr_state_dict = vae.state_dict()
        #curr_state_dict.update(loaded_state_dict)
        vae.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)

    if args.eval:
        logging('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)

            test(vae, test_data_batch, test_labels_batch, "TEST", args)
            au, au_var = calc_au(vae, test_data_batch)
            logging("%d active units" % au)
            # print(au_var)

            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            calc_iwnll(vae, test_data_batch, args)

        return

    if args.discriminator == "linear":
        discriminator = LinearDiscriminator(args, vae.encoder).to(device)
    elif args.discriminator == "mlp":
        discriminator = MLPDiscriminator(args, vae.encoder).to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=args.momentum)
        opt_dict['lr'] = args.lr
    elif args.opt == "adam":
        optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        opt_dict['lr'] = 0.001
    else:
        raise ValueError("optimizer not supported")

    iter_ = decay_cnt = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    pre_mi = 0
    discriminator.train()
    start = time.time()

    kl_weight = args.kl_start
    if args.warm_up > 0:
        anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))
    else:
        anneal_rate = 0

    dim_target_kl = args.target_kl / float(args.nz)

    train_data_batch, train_labels_batch = train_data.create_data_batch_labels(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch, val_labels_batch = val_data.create_data_batch_labels(batch_size=128,
                                                device=device,
                                                batch_first=True)

    test_data_batch, test_labels_batch = test_data.create_data_batch_labels(batch_size=128,
                                                  device=device,
                                                  batch_first=True)

    acc_cnt = 1
    acc_loss = 0.
    for epoch in range(args.epochs):
        report_loss = 0
        report_correct = report_num_words = report_num_sents = 0
        acc_batch_size = 0
        optimizer.zero_grad()
        for i in np.random.permutation(len(train_data_batch)):

            batch_data = train_data_batch[i]
            batch_labels = train_labels_batch[i]
            batch_labels = [int(x) for x in batch_labels]

            batch_labels = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=device)

            batch_size, sent_len = batch_data.size()

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size
            acc_batch_size += batch_size

            # (batch_size)
            loss, correct = discriminator.get_performance(batch_data, batch_labels)

            acc_loss = acc_loss + loss.sum()

            if acc_cnt % args.update_every == 0:
                acc_loss = acc_loss / acc_batch_size
                acc_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)

                optimizer.step()
                optimizer.zero_grad()

                acc_cnt = 0
                acc_loss = 0
                acc_batch_size = 0

            acc_cnt += 1
            report_loss += loss.sum().item()
            report_correct += correct

            if iter_ % log_niter == 0:
                #train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
                train_loss = report_loss / report_num_sents

        
                logging('epoch: %d, iter: %d, avg_loss: %.4f, acc %.4f,' \
                       'time %.2fs' %
                       (epoch, iter_, train_loss, report_correct / report_num_sents,
                        time.time() - start))

                #sys.stdout.flush()

            iter_ += 1

        logging('lr {}'.format(opt_dict["lr"]))

        discriminator.eval()

        with torch.no_grad():
            loss, acc = test(discriminator, val_data_batch, val_labels_batch, "VAL", args)
            # print(au_var)

        if loss < best_loss:
            logging('update best loss')
            best_loss = loss
            best_acc = acc
            torch.save(discriminator.state_dict(), args.save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >= args.load_best_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                discriminator.load_state_dict(torch.load(args.save_path))
                logging('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                if args.opt == "sgd":
                    optimizer = optim.SGD(discriminator.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    opt_dict['lr'] = opt_dict["lr"]
                elif args.opt == "adam":
                    optimizer = optim.Adam(discriminator.parameters(), lr=opt_dict["lr"])
                    opt_dict['lr'] = opt_dict["lr"]
                else:
                    raise ValueError("optimizer not supported")                

        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, acc = test(discriminator, test_data_batch, test_labels_batch, "TEST", args)

        discriminator.train()


    # compute importance weighted estimate of log p(x)
    discriminator.load_state_dict(torch.load(args.save_path))
    discriminator.eval()

    with torch.no_grad():
        loss, acc = test(discriminator, test_data_batch, test_labels_batch, "TEST", args)
        # print(au_var)

if __name__ == '__main__':
    args = init_config()
    main(args)
