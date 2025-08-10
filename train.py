import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import roc_auc_score, pearsonr

from model.deepbind import DeepBind
from data.pbm import PBMDataset
from data.rnacompete import RNAcompeteDataset
from data.chipseq import ChIPSeqDataset

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
    
def build_dataset(args):
    if args.data_type == 'PBM':
        return PBMDataset(args.seq_file, args.tgt_file, args.tf_col, args.max_len)
    elif args.data_type == 'RNAcompete':
        return RNAcompeteDataset(args.raw_file, args.max_len)
    elif args.data_type == 'ChIPSeq':
        return ChIPSeqDataset(args.peaks_bed, args.fasta_file, flank=(args.max_len - 1) // 2)
    else:
        raise ValueError("Unsupported data type")
    
def sample_hyperparams():
    return{
        'lr': 10 ** random.uniform(-4, -2),
        'weight_decay': 10 ** random.uniform(-6, -3),
        'batch_size': random.choice([32, 64, 128]),
        'num_kernels': random.choice([8, 16, 32]),
        'kernel_size': random.choice([12, 24, 36]),
        'fc_hidden': random.choice([16, 32, 64]),
    }
    
def cv_score(dataset, hp, device, binary=False):
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    fold_size = n // 3
    metrics = []
    for k in range(3):
        val_idx = idxs[k * fold_size : (k + 1) * fold_size]
        train_idx = [i for i in idxs if i not in val_idx]
        train_dl = DataLoader(Subset(dataset, train_idx),
                              batch_size=hp['batch_size'], shuffle=True)
        val_dl = DataLoader(Subset(dataset, val_idx),
                            batch_size=hp['batch_size'])
        model = DeepBind(
            num_kernels=hp['num_kernels'],
            kernel_size=hp['kernel_size'],
            fc_hidden=hp['fc_hidden'].to(device),
        )
        opt = torch.optim.Adam(
            model.parameters(),
            lr=hp['lr'],
            weight_decay=hp['weight_decay'],
        )
        loss_fn = (nn.BCEWithLogitsLoss() if binary else nn.MSELoss())
        model.train()
        for _ in range(3):
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                lgt = model(xb)
                all_logits.append(lgt.cpu())
                all_labels.append(yb.cpu())
        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        if binary:
            probs = 1 / (1 + np.exp(-logits))
            metrics.append(roc_auc_score(labels, probs))
        else:
            metrics.append(pearsonr(logits, labels)[0])
    return float(np.mean(metrics))

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    dataset = build_dataset(args)
    n_test = int(len(dataset) * args.test_frac)
    n_train = len(dataset) - n_test
    rest_ds, test_ds = random_split(dataset, [n_train, n_test])
    test_dl = DataLoader(test_ds, batch_size=args.batch_size if hasattr(args, 'batch_size') else 64)
    
    best_hp, best_score = None, -np.inf
    for i in range(30):
        hp = sample_hyperparams()
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
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp['lr'],
        weight_decay=hp['weight_decay']
    )
    loss_fn = nn.BCEWithLogitsLoss() if args.data_type == 'PBM' else nn.MSELoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        print(f"[Train] Epoch{epoch:02d} loss={total_loss/len(rest_ds):.4f}")
        
    torch.save(model.state_dict(), "deepbind_final.pth")
    
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb,yb in test_dl:
            xb,yb = xb.to(device), yb.to(device)
            all_logits.append(model(xb).cpu())
            all_labels.append(yb.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    pear, _ = pearsonr(logits, labels)
    print(f">> Test Pearson: {pear:.4f}")

if __name__ == '__main__':
    main()