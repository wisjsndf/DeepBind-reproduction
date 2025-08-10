import torch

BASE2IDX = {'A':0, 'C':1, 'G':2, 'T':3}

def one_hot_encode(sequence: str, max_len: int = 1000):
    mat = torch.zeros(4, max_len, dtype=torch.float32)
    for i, base in enumerate(sequence[:max_len]):
        idx = BASE2IDX.get(base)
        if idx is not None:
            mat[idx, i] = 1.0
    return mat