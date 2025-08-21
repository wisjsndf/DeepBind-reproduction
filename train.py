import os
import argparse
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from model.deepbind import DeepBind
from data.pbm import PBMDataset
from data.rnacompete import RNAcompeteDataset
from data.chipseq import ChIPSeqDataset

def log_uniform(a, b):
    return 10 ** random.uniform(math.log10(a), math.log10(b))

def sqrt_uniform(a, b):
    u = random.uniform(a ** 2, b ** 2)
    return math.sqrt(u)

def steps_to_epochs(num_steps, batch_size, train_size):
    if train_size <= 0:
        return 1
    epochs = math.ceil((num_steps * batch_size) / float(train_size))
    return int(max(1, epochs))

def safe_pearson(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float('nan')
    # check constant arrays (all elements equal)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float('nan')
    try:
        r = pearsonr(a, b)[0]
        if np.isnan(r):
            return float('nan')
        return float(r)
    except Exception:
        return float('nan')

def apply_initialization(model, hp):
    return

def prepare_batch(xb, yb, device):
    xb = xb.to(device).float()
    yb = yb.to(device).float()
    if yb.dim() == 1:
        yb = yb.unsqueeze(1)
    return xb, yb
    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_type', choices=['PBM', 'RNAcompete', 'ChIPSeq'], required=True)
    p.add_argument('--seq_file', type=str, help="PBM: sequences.tsv")
    p.add_argument('--tgt_file', type=str, help="PBM: targets.tsv")
    p.add_argument('--tf_col', type=str, help="PBM: target column")
    p.add_argument('--raw_file', type=str, help="RNAcompete: raw txt/tsv")
    p.add_argument('--peaks_bed', type=str, help="ChIP-seq peaks.bed")
    p.add_argument('--fasta_file', type=str, help="ChIP-seq genome.fa")
    p.add_argument('--max_len', type=int, default=36)
    p.add_argument('--test_frac', type=float, default=0.1, help="portion of data to use for testing")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()
    
def build_dataset(args):
    if args.data_type == 'PBM':
        return PBMDataset(args.seq_file, args.tgt_file, args.tf_col, args.max_len)
    elif args.data_type == 'RNAcompete':
        return RNAcompeteDataset(args.raw_file, args.max_len)
    elif args.data_type == 'ChIPSeq':
        if args.seq_file:
            return ChIPSeqDataset(seq_file=args.seq_file, max_len=args.max_len, sep='\t')
        else:
            return ChIPSeqDataset(peaks_bed=args.peaks_bed, fasta_file=args.fasta_file, flank=(args.max_len-1)//2)
    else:
        raise ValueError("Unsupported data type")

def sample_hyperparams(dataset='DREAM5'):
    dataset_l = dataset.lower() if isinstance(dataset, str) else 'dream5'
    if dataset_l in ('dream5', 'encode'):
        motif_length = 24
    elif dataset_l in ('rnacompete', 'pbm'):
        motif_length = 16
    elif dataset_l == 'chipseq':
        motif_length = 36
    else:
        motif_length = 24
    
    num_motifs = 16
    hidden = random.choice([None, 32])
    
    lr = log_uniform(5e-4, 5e-2)
    momentum = sqrt_uniform(0.95, 0.99)
    batch_size = 64
    checkpoint_steps = [4000, 8000, 12000, 16000, 20000]
    
    init_scale_motifs = log_uniform(1e-7, 1e-3)
    init_scale_nn = log_uniform(1e-5, 1e-2)
    
    weight_decay_motifs = log_uniform(1e-15, 1e-3)
    weight_decay_nn = log_uniform(1e-10, 1e-3)
    
    keep_prob = random.choice([0.5, 0.75, 1.0])
    dropout_rate = 1.0 - keep_prob
    
    return {
        'num_kernels': int(num_motifs),
        'kernel_size': int(motif_length),
        'fc_hidden': int(hidden) if hidden is not None else 0,
        'lr': float(lr),
        'momentum': float(momentum),
        'batch_size': int(batch_size),
        'checkpoint_steps': checkpoint_steps,
        'init_scale_motifs': float(init_scale_motifs),
        'init_scale_nn': float(init_scale_nn),
        'weight_decay_motifs': float(weight_decay_motifs),
        'weight_decay_nn': float(weight_decay_nn),
        'dropout_rate': float(dropout_rate),
    }
    
def cv_score(dataset, hp, device, binary=False):
    n = len(dataset)
    if n < 3:
        return float('nan')
    idxs = list(range(n))
    random.shuffle(idxs)
    fold_size = n // 3
    fold_scores = []
    for k in range(3):
        start = k * fold_size
        end = (k + 1) * fold_size if k < 2 else n
        val_idx = idxs[start:end]
        train_idx = [i for i in idxs if i not in val_idx]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue

        train_dl = DataLoader(Subset(dataset, train_idx),
                              batch_size=hp['batch_size'], shuffle=True)
        val_dl = DataLoader(Subset(dataset, val_idx),
                            batch_size=hp['batch_size'], shuffle=False)

        # build model and init
        model = DeepBind(
            num_kernels=hp['num_kernels'],
            kernel_size=hp['kernel_size'],
            fc_hidden=hp['fc_hidden']
        ).to(device)
        apply_initialization(model, hp)

        # param groups for separate weight_decay
        conv_params = []
        nn_params = []
        for name, p in model.named_parameters():
            if ('conv' in name.lower()) or ('kernel' in name.lower()) or ('motif' in name.lower()):
                conv_params.append(p)
            else:
                nn_params.append(p)
        param_groups = []
        if len(conv_params) > 0:
            param_groups.append({'params': conv_params, 'weight_decay': hp['weight_decay_motifs']})
        if len(nn_params) > 0:
            param_groups.append({'params': nn_params, 'weight_decay': hp['weight_decay_nn']})
        if len(param_groups) == 0:
            param_groups = [{'params': model.parameters(), 'weight_decay': (hp['weight_decay_nn'] + hp['weight_decay_motifs']) / 2.0}]

        opt = torch.optim.SGD(param_groups, lr=hp['lr'], momentum=hp['momentum'], nesterov=True)

        loss_fn = (nn.BCEWithLogitsLoss() if binary else nn.MSELoss())

        train_size = len(train_idx)
        checkpoint_steps = hp.get('checkpoint_steps', [4000,8000,12000,16000,20000])
        epoch_targets = [steps_to_epochs(s, hp['batch_size'], train_size) for s in checkpoint_steps]
        epoch_targets = sorted(list(dict.fromkeys(epoch_targets)))

        best_val_score = -np.inf
        current_epoch = 0
        for target_epoch in epoch_targets:
            epochs_to_run = target_epoch - current_epoch
            if epochs_to_run > 0:
                model.train()
                for e in range(epochs_to_run):
                    for xb, yb in train_dl:
                        xb, yb = prepare_batch(xb, yb, device)
                        out = model(xb)
                        if out.dim() == 1:
                            out = out.unsqueeze(1)
                        loss = loss_fn(out, yb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                current_epoch = target_epoch

            # evaluate on validation
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = prepare_batch(xb, yb, device)
                    lgt = model(xb)
                    if lgt.dim() == 1:
                        lgt = lgt.unsqueeze(1)
                    all_logits.append(lgt.cpu())
                    all_labels.append(yb.cpu())
            if len(all_logits) == 0:
                continue
            logits = torch.cat(all_logits).numpy().reshape(-1)
            labels = torch.cat(all_labels).numpy().reshape(-1)
            if binary:
                probs = 1 / (1 + np.exp(-logits))
                try:
                    score = roc_auc_score(labels, probs)
                except Exception:
                    score = float('nan')
            else:
                score = safe_pearson(logits, labels)
            if np.isnan(score):
                print(f"[DEBUG] Fold {k} got NaN score")
                print(f" labels: mean={labels.mean():.4f}, std={labels.std():.4f}, unique={np.unique(labels)[:10]}")
                print(f" preds : mean={logits.mean():.4f}, std={logits.std():.4f}, min={logits.min():.4f}, max={logits.max():.4f}")

            if not np.isnan(score) and score > best_val_score:
                best_val_score = score

        if best_val_score == -np.inf:
            fold_scores.append(float('nan'))
        else:
            fold_scores.append(best_val_score)

    valid = [s for s in fold_scores if not (s is None or np.isnan(s))]
    if len(valid) == 0:
        return float('nan')
    return float(np.mean(valid))

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    dataset = build_dataset(args)
    n_test = int(len(dataset) * args.test_frac)
    n_train = len(dataset) - n_test
    rest_ds, test_ds = random_split(dataset, [n_train, n_test])
    
    best_hp, best_score = None, -np.inf
    for i in range(30):
        hp = sample_hyperparams(dataset='DREAM5')
        score = cv_score(rest_ds, hp, device, binary=False)
        print(f"[Calib {i+1:02d}/30] hp={hp} CV_score={score:.4f}")
        if score > best_score:
            best_hp, best_score = hp, score
    print(f"Best hyperparams: {best_hp} with CV score {best_score:.4f}")
    
    hp = best_hp
    train_dl = DataLoader(rest_ds, batch_size=hp['batch_size'], shuffle=True)
    model = DeepBind(
        num_kernels=hp['num_kernels'],
        kernel_size=hp['kernel_size'],
        fc_hidden=hp['fc_hidden'],
    ).to(device)
    apply_initialization(model, hp)
    
    conv_params = []
    nn_params = []
    for name, p in model.named_parameters():
        if ('conv' in name.lower()) or ('kernel' in name.lower()) or ('motif' in name.lower()):
            conv_params.append(p)
        else:
            nn_params.append(p)
    param_groups = []
    if len(conv_params) > 0:
        param_groups.append({'params': conv_params, 'weight_decay': hp['weight_decay_motifs']})
    if len(nn_params) > 0:
        param_groups.append({'params': nn_params, 'weight_decay': hp['weight_decay_nn']})
    if len(param_groups) == 0:
        param_groups = [{'params': model.parameters(), 'weight_decay': (hp['weight_decay_nn'] + hp['weight_decay_motifs']) / 2.0}]
        
    opt = torch.optim.SGD(param_groups, lr=hp['lr'], momentum=hp['momentum'], nesterov=True)
    loss_fn = nn.BCEWithLogitsLoss() if args.data_type == 'ChIPSeq' else nn.MSELoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = prepare_batch(xb, yb, device)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"[Train] Epoch{epoch:02d} loss={total_loss/len(rest_ds):.4f}")
        
    torch.save(model.state_dict(), "deepbind_final.pth")
    
    test_dl = DataLoader(test_ds, batch_size=hp['batch_size'])
    
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb,yb in test_dl:
            xb,yb = prepare_batch(xb, yb, device)
            lgt = model(xb)
            if lgt.dim() == 1:
                lgt = lgt.unsqueeze(1)
            all_logits.append(model(xb).cpu())
            all_labels.append(yb.cpu())
    logits = torch.cat(all_logits).numpy().reshape(-1)
    labels = torch.cat(all_labels).numpy().reshape(-1)
    pear = safe_pearson(logits, labels)
    if np.isnan(pear):
        print(">> Test Pearson: nan (constant preds or labels; pearson undefined)")
    else:
        print(f">> Test Pearson: {pear:.4f}")

if __name__ == '__main__':
    main()