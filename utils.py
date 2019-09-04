import numpy as np
import os, sys
import torch
from torch import nn, optim
import subprocess

class uniform_initializer(object):
        def __init__(self, stdv):
            self.stdv = stdv
        def __call__(self, tensor):
            nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

def reconstruct(model, test_data_batch, vocab, strategy, fname):
    hyps = []
    refs = []
    with open(fname, "w") as fout:
        #for i in range(10):
            # batch_data = test_data_batch[i]

        for batch_data in test_data_batch:
            decoded_batch = model.reconstruct(batch_data, strategy)

            source = [[vocab.id2word(id_.item()) for id_ in sent] for sent in batch_data]
            for j in range(len(batch_data)):
                ref = " ".join(source[j])
                hyp = " ".join(decoded_batch[j])
                fout.write("SOURCE: {}\n".format(ref))
                fout.write("RECON: {}\n\n".format(hyp))

                refs += [ref[len("<s>"): -len("</s>")]]
                if strategy == "beam":
                    hyps += [hyp[len("<s>"): -len("</s>")]]
                else:
                    hyps += [hyp[: -len("</s>")]]

    fname_ref = fname + ".ref"
    fname_hyp = fname + ".hyp"
    with open(fname_ref, "w") as f:
        f.write("\n".join(refs))
    with open(fname_hyp, "w") as f:
        f.write("\n".join(hyps))
    call_multi_bleu_perl("scripts/multi-bleu.perl", fname_hyp, fname_ref, verbose=True)


def calc_iwnll(model, test_data_batch, args, ns=100):

    report_nll_loss = 0
    report_num_words = report_num_sents = 0
    print("iw nll computing ", end="")
    for id_, i in enumerate(np.random.permutation(len(test_data_batch))):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size
        if id_ % (round(len(test_data_batch) / 20)) == 0:
            print('%d%% ' % (id_/(round(len(test_data_batch) / 20)) * 5), end="")
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples, ns=ns)

        report_nll_loss += loss.sum().item()

    print()
    sys.stdout.flush()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    return nll, ppl

# def calc_mi(model, test_data_batch):
#     mi = 0
#     num_examples = 0
#     for batch_data in test_data_batch:
#         batch_size = batch_data.size(0)
#         num_examples += batch_size
#         mutual_info = model.calc_mi_q(batch_data)
#         mi += mutual_info * batch_size

#     return mi / num_examples

def calc_mi(model, test_data_batch):
    # calc_mi_v3
    import math 
    from modules.utils import log_sum_exp

    mi = 0
    num_examples = 0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    for batch_data in test_data_batch:
        mu, logvar = model.encoder.forward(batch_data)
        x_batch, nz = mu.size()
        ##print(x_batch, end=' ')
        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy += (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]

    neg_entropy = neg_entropy / num_examples
    ##print()

    num_examples = 0
    log_qz = 0.
    for i in range(len(mu_batch_list)):
        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        
        # [z_batch, 1, nz]
        if hasattr(model.encoder, 'reparameterize'):
            z_samples = model.encoder.reparameterize(mu, logvar, 1)
        else:
            z_samples = model.encoder.gaussian_enc.reparameterize(mu, logvar, 1)
        z_samples = z_samples.view(-1, 1, nz)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]
        #mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
        #indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi


def calc_au(model, test_data_batch, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item(), au_var


def sample_sentences(vae, vocab, device, num_sentences):
    global logging

    vae.eval()
    sampled_sents = []
    for i in range(num_sentences):
        z = vae.sample_from_prior(1)
        z = z.view(1,1,-1)
        start = vocab.word2id['<s>']
        # START = torch.tensor([[[start]]])
        START = torch.tensor([[start]])
        end = vocab.word2id['</s>']
        START = START.to(device)
        z = z.to(device)
        vae.eval()
        sentence = vae.decoder.sample_text(START, z, end, device)
        decoded_sentence = vocab.decode_sentence(sentence)
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging(i,":",' '.join(sent))

# def visualize_latent(args, vae, device, test_data):
#     f = open('yelp_embeddings_z','w')
#     g = open('yelp_embeddings_labels','w')

#     test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
#     for i in range(len(test_data_batch)):
#         batch_data = test_data_batch[i]
#         batch_label = test_label_batch[i]
#         batch_size, sent_len = batch_data.size()
#         means, _ = vae.encoder.forward(batch_data)
#         for i in range(batch_size):
#             mean = means[i,:].cpu().detach().numpy().tolist()
#             for val in mean:
#                 f.write(str(val)+'\t')
#             f.write('\n')
#         for label in batch_label:
#             g.write(label+'\n')
#         fo
#         print(mean.size())
#         print(logvar.size())
#         fooo

def visualize_latent(args, epoch, vae, device, test_data):
    nsamples = 1

    with open(os.path.join(args.exp_dir, f'synthetic_latent_{epoch}.txt'),'w') as f:
        test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size, device=device, batch_first=True)
        for i in range(len(test_data_batch)):
            batch_data = test_data_batch[i]
            batch_label = test_label_batch[i]
            batch_size, sent_len = batch_data.size()
            samples, _ = vae.encoder.encode(batch_data, nsamples)
            for i in range(batch_size):
                for j in range(nsamples):
                    sample = samples[i,j,:].cpu().detach().numpy().tolist()
                    f.write(batch_label[i] + '\t' + ' '.join([str(val) for val in sample]) + '\n')


def call_multi_bleu_perl(fname_bleu_script, fname_hyp, fname_ref, verbose=True):
    cmd = "perl %s %s < %s" % (fname_bleu_script, fname_ref, fname_hyp)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, \
        stderr=subprocess.PIPE, shell=True)
    popen.wait()
    try:
        bleu_result = popen.stdout.readline().strip().decode("utf-8")
        if verbose:
            print(bleu_result)
        bleu = float(bleu_result[7:bleu_result.index(',')])
        stderrs = popen.stderr.readlines()
        if len(stderrs) > 1:
            for line in stderrs:
                print(line.strip())
    except Exception as e:
        print(e)
        bleu = 0.
    return bleu

































