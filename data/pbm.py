# data/pbm.py
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import one_hot_encode

class PBMDataset(Dataset):
    def __init__(self, seq_file, target_file, tf_col, max_len=36, preload=True, compression='infer'):
        """
        seq_file: tsv or tsv.gz containing sequences (try column named 'seq' or last column)
        target_file: tsv or tsv.gz with header; tf_col is column name or index
        max_len: sequence length (pad/truncate)
        preload: if True, precompute all one-hot arrays into memory (fast at training)
        """
        self.max_len = int(max_len)
        self.preload = bool(preload)

        # read sequences robustly
        try:
            df_seq = pd.read_csv(seq_file, sep='\t', header=0, compression=compression, engine='python')
            seq_col = None
            for c in df_seq.columns:
                if str(c).lower() == 'seq':
                    seq_col = c
                    break
            if seq_col is None:
                seq_col = df_seq.columns[-1]
            seqs = df_seq[seq_col].astype(str).tolist()
        except Exception:
            df_seq = pd.read_csv(seq_file, sep='\t', header=None, compression=compression, engine='python')
            seqs = df_seq.iloc[:, -1].astype(str).tolist()

        # read targets
        tgt_df = pd.read_csv(target_file, sep='\t', header=0, compression=compression, engine='python')
        if isinstance(tf_col, int) or (isinstance(tf_col, str) and tf_col.isdigit()):
            col_idx = int(tf_col)
            if col_idx < 0 or col_idx >= tgt_df.shape[1]:
                raise ValueError(f"tf_col index {col_idx} out of range for {target_file}")
            labels_all = tgt_df.iloc[:, col_idx].astype(float).values
        else:
            if tf_col not in tgt_df.columns:
                raise ValueError(f"tf_col '{tf_col}' not found. Available: {list(tgt_df.columns)[:50]}")
            labels_all = tgt_df[tf_col].astype(float).values

        # align lengths
        n_seq = len(seqs)
        n_lbl = len(labels_all)
        if n_seq != n_lbl:
            m = min(n_seq, n_lbl)
            warnings.warn(f"seqs ({n_seq}) != labels ({n_lbl}) -> trimming to {m}")
            seqs = seqs[:m]
            labels_all = labels_all[:m]

        # keep only non-NaN labels
        valid_mask = ~np.isnan(labels_all)
        seqs = [s.upper() for s, ok in zip(seqs, valid_mask) if ok]
        labels = labels_all[valid_mask].astype(np.float32)

        self.n = len(labels)
        self.labels = labels
        self.seqs = seqs
        self.label_mean = float(np.mean(self.labels))
        self.label_std = float(np.std(self.labels)) if float(np.std(self.labels)) > 0.0 else 1.0
        self.labels_norm = ((self.labels - self.label_mean) / self.label_std).astype(np.float32)
        # preload one-hot encodings into memory if requested
        if self.preload:
            X = np.zeros((self.n, 4, self.max_len), dtype=np.float32)
            for i, s in enumerate(self.seqs):
                arr = one_hot_encode(s, self.max_len)
                # ensure arr is (4, max_len)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.shape != (4, self.max_len):
                    if arr.shape[0] == self.max_len and arr.shape[1] == 4:
                        arr = arr.T
                    else:
                        tmp = np.zeros((4, self.max_len), dtype=np.float32)
                        min_r = min(arr.shape[0], 4)
                        min_c = min(arr.shape[1], self.max_len)
                        tmp[:min_r, :min_c] = arr[:min_r, :min_c]
                        arr = tmp
                X[i] = arr
            self.X = X
        else:
            self.X = None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.preload and self.X is not None:
            x = torch.from_numpy(self.X[idx])  # float32
        else:
            s = self.seqs[idx]
            arr = one_hot_encode(s, self.max_len)
            arr = np.asarray(arr, dtype=np.float32)
            x = torch.tensor(arr, dtype=torch.float32)
        y = torch.tensor(float(self.labels_norm[idx]), dtype=torch.float32)
        return x, y
