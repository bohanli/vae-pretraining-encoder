import argparse
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

parser = argparse.ArgumentParser(description='GMM unsupervised clustering')
parser.add_argument('--exp_dir', type=str)
parser.add_argument('--num', type=int, default=10)
parser.add_argument('--pca_num', type=int, default=0)
parser.add_argument('--one2one', action="store_true", default=False)

args = parser.parse_args()

gmm = GaussianMixture(n_components=args.num, tol=1e-3, max_iter=200, n_init=1, verbose=1)

if args.pca_num > 0:
    pca = PCA(n_components=args.pca_num)


train_x = np.loadtxt(os.path.join(args.exp_dir, "train.vec"), delimiter="\t")
valid_x = np.loadtxt(os.path.join(args.exp_dir, "val.vec"), delimiter="\t")
test_x = np.loadtxt(os.path.join(args.exp_dir, "test.vec"), delimiter="\t")

if args.pca_num > 0:
    pca.fit(train_x)

    train_x = pca.transform(train_x)
    valid_x = pca.transform(valid_x)
    test_x = pca.transform(test_x)

print(train_x.shape)

print("start fitting gmm on training data")
gmm.fit(train_x)

valid_pred_y = gmm.predict(valid_x)
valid_true_y = np.loadtxt(os.path.join(args.exp_dir, "val.label"), dtype=np.int)

if args.one2one:
    print("linear assignment")
    cost_matrix = np.zeros((args.num, args.num))

    for i, j in zip(valid_pred_y, valid_true_y):
        cost_matrix[i,j] -= 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
else:
    # (nsamples, ncomponents)
    valid_score = gmm.predict_proba(valid_x)
    valid_max_index = np.argmax(valid_score, axis=0)
    col_ind = {}
    for i in range(args.num):
        col_ind[i] = valid_true_y[valid_max_index[i]]

print(col_ind)
correct = 0.
for i, j in zip(valid_pred_y, valid_true_y):
    if col_ind[i] == j:
        correct += 1
print("validation acc {}".format(correct / len(valid_pred_y)))

test_pred_y = gmm.predict(test_x)
test_true_y = np.loadtxt(os.path.join(args.exp_dir, "test.label"), dtype=np.int)
correct = 0.
for i, j in zip(test_pred_y, test_true_y):
    if col_ind[i] == j:
        correct += 1
print("test acc {}".format(correct / len(test_pred_y)))

train_pred_y = gmm.predict(train_x)
train_true_y = np.loadtxt(os.path.join(args.exp_dir, "train.label"), dtype=np.int)
correct = 0.
for i, j in zip(train_pred_y, train_true_y):
    if col_ind[i] == j:
        correct += 1
print("train acc {}".format(correct / len(train_pred_y)))
