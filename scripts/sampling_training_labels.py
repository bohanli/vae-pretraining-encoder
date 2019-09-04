"""
This script samples a certain number of training samples for fine-tuning
"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='sampling')
parser.add_argument('--num_label', type=int)
parser.add_argument('--seed', type=int, default="783435")
parser.add_argument('--split_even', action="store_true", default=False)
parser.add_argument('--dataset', type=str)

args = parser.parse_args()

np.random.seed(args.seed)

data_dir = os.path.join("datasets/{}_data".format(args.dataset))
fout = open(os.path.join(data_dir, "{}.train.{}.txt".format(args.dataset, args.num_label)), "w")

if args.split_even:
    label2text = {}
    with open(os.path.join(data_dir, "{}.train.txt".format(args.dataset))) as fin:
        for line in fin:
            label, text = line.strip().split("\t")
            label = int(label)

            if label in label2text:
                label2text[label] += [text]
            else:
                label2text[label] = [text]


    nlabel = len(label2text)
    sample_per_label = int(args.num_label / nlabel)
    for i in range(nlabel):
        index = np.random.choice(len(label2text[i]), sample_per_label, replace=False)
        for j in index:
            fout.write("{}\t{}\n".format(i, label2text[i][j]))

else:
    with open(os.path.join(data_dir, "{}.train.txt".format(args.dataset))) as fin:
        text = fin.readlines()
        index = np.random.choice(len(text), args.num_label, replace=False)
        for i in index:
            fout.write(text[i])

fout.close()    