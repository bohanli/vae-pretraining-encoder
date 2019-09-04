import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LinearDiscriminator(nn.Module):
    """docstring for LinearDiscriminator"""
    def __init__(self, args, encoder):
        super(LinearDiscriminator, self).__init__()
        self.args = args

        self.encoder = encoder
        if args.freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(args.nz, args.ncluster)
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def get_performance(self, batch_data, batch_labels):
        mu, _ = self.encoder(batch_data)
        if not self.args.freeze_enc:
            mu = self.dropout(mu)
        logits = self.linear(mu)
        loss = self.loss(logits, batch_labels)

        _, pred = torch.max(logits, dim=1)
        correct = torch.eq(pred, batch_labels).float().sum().item()

        return loss, correct


class MLPDiscriminator(nn.Module):
    """docstring for LinearDiscriminator"""
    def __init__(self, args, encoder):
        super(MLPDiscriminator, self).__init__()
        self.args = args

        self.encoder = encoder
        if args.freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.feats = nn.Sequential(
            nn.Linear(args.nz, args.nz),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.nz, args.nz),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.nz, args.ncluster),
        )
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def get_performance(self, batch_data, batch_labels):
        mu, _ = self.encoder(batch_data)
        logits = self.feats(mu)
        loss = self.loss(logits, batch_labels)

        _, pred = torch.max(logits, dim=1)
        correct = torch.eq(pred, batch_labels).float().sum().item()

        return loss, correct
