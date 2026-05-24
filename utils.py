import os
import random
from math import inf

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from scipy import stats
from scipy.io.arff import loadarff
from sklearn.metrics import accuracy_score, f1_score


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transfer_labels(labels):
    indices = np.unique(labels)
    for i in range(labels.shape[0]):
        labels[i] = np.argwhere(labels[i] == indices)[0][0]
    return labels


def build_dataset_uea(args):
    data_path = os.path.join(args.data_path, args.dataset)
    train_raw = loadarff(os.path.join(data_path, args.dataset + '_TRAIN.arff'))[0]
    test_raw  = loadarff(os.path.join(data_path, args.dataset + '_TEST.arff'))[0]

    def extract(raw):
        xs, ys = [], []
        for row_x, row_y in raw:
            xs.append(np.array([d.tolist() for d in row_x]))
            ys.append(row_y.decode('utf-8'))
        # shape: (N, channels, time)
        return np.array(xs).swapaxes(1, 2).transpose(0, 2, 1), np.array(ys)

    X_tr, y_tr = extract(train_raw)
    X_te, y_te = extract(test_raw)

    lbl_map = {k: i for i, k in enumerate(np.unique(y_tr))}
    y_tr = np.vectorize(lbl_map.get)(y_tr).astype(np.int64)
    y_te = np.vectorize(lbl_map.get)(y_te).astype(np.int64)

    for arr in (X_tr, X_te):
        idx = np.where(np.isnan(arr))
        col_mean = np.nanmean(arr, axis=0)
        col_mean[np.isnan(col_mean)] = 1e-6
        arr[idx] = np.take(col_mean, idx[1])

    return X_tr, y_tr, X_te, y_te, len(lbl_map)


def build_dataset_ucr(args):
    root = os.path.join(args.data_path, args.dataset)
    tr = pd.read_csv(os.path.join(root, args.dataset + '_TRAIN.tsv'), sep='\t', header=None)
    te = pd.read_csv(os.path.join(root, args.dataset + '_TEST.tsv'),  sep='\t', header=None)

    X_tr = torch.unsqueeze(torch.from_numpy(tr.iloc[:, 1:].to_numpy(np.float32)), 1).numpy()
    X_te = torch.unsqueeze(torch.from_numpy(te.iloc[:, 1:].to_numpy(np.float32)), 1).numpy()
    y_tr = transfer_labels(tr.iloc[:, 0].to_numpy(np.float32))
    y_te = transfer_labels(te.iloc[:, 0].to_numpy(np.float32))

    for arr in (X_tr, X_te):
        idx = np.where(np.isnan(arr))
        col_mean = np.nanmean(arr, axis=0)
        col_mean[np.isnan(col_mean)] = 1e-6
        arr[idx] = np.take(col_mean, idx[1])

    return X_tr, y_tr, X_te, y_te, len(np.unique(y_tr))


class TimeDatasetWithIndex(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index], index

    def __len__(self):
        return len(self.target)


def shuffler(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std=0.1, seed=42):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    flip_dist = stats.truncnorm(
        (0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std
    )
    flip_rate = flip_dist.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()
    W = torch.FloatTensor(np.random.randn(num_classes, feature_size, num_classes)).cuda()

    P = []
    for x, y in dataset:
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[len(P)] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[len(P)]
        P.append(A)

    P = torch.stack(P).cpu().numpy()
    new_label = [np.random.choice(num_classes, p=P[i]) for i in range(len(P))]
    print(f'Instance noise rate = {(np.array(new_label) != labels.cpu().numpy()).mean():.4f}')
    return np.array(new_label)


def flip_label(dataset, target, ratio, args, pattern=0):
    """
    pattern:  0  = symmetric,  1 = asymmetric,  -1 = instance-dependent
    Returns (noisy_labels, mask)  where mask[i]=1 means sample i was flipped.
    """
    assert 0 <= ratio < 1
    target = np.array(target, dtype=int)
    label  = target.copy()
    n_class = len(np.unique(label))

    if pattern == -1:
        data_t    = torch.from_numpy(dataset).float()
        targets_t = torch.from_numpy(target).long()
        label = get_instance_noisy_label(
            n=ratio, dataset=zip(data_t, targets_t), labels=targets_t,
            num_classes=n_class,
            feature_size=dataset.shape[1] * dataset.shape[2],
            seed=args.random_seed
        )
    else:
        for i in range(len(label)):
            if pattern == 0:
                p = np.ones(n_class) * ratio / (n_class - 1)
                p[label[i]] = 1 - ratio
                label[i] = np.random.choice(n_class, p=p)
            elif pattern == 1:
                label[i] = np.random.choice(
                    [label[i], (target[i] + 1) % n_class], p=[1 - ratio, ratio]
                )

    mask = (label != target).astype(int)
    return label, mask


def evaluate_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    return accuracy_score(targets, preds), f1_score(targets, preds, average='macro')


def adjust_learning_rate(optimizer, epoch, lr0, total_epochs, warmup_epochs=10):
    if epoch < warmup_epochs:
        lr = lr0 * (epoch + 1) / warmup_epochs
    else:
        lr = lr0 * 0.5 * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., ICLR 2021)."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        g_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (g_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        dev = self.param_groups[0]['params'][0].device
        return torch.norm(torch.stack([
            ((torch.abs(p) if g['adaptive'] else 1.0) * p.grad).norm(2).to(dev)
            for g in self.param_groups for p in g['params'] if p.grad is not None
        ]), p=2)
