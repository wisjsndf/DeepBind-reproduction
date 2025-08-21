import torch
import numpy as np

# build lookup table once (ASCII -> index)
_LU = np.full(256, 4, dtype=np.uint8)   # default 4 => unknown
_LU[ord('A')] = 0; _LU[ord('C')] = 1; _LU[ord('G')] = 2; _LU[ord('T')] = 3
_LU[ord('a')] = 0; _LU[ord('c')] = 1; _LU[ord('g')] = 2; _LU[ord('t')] = 3

def one_hot_encode(seq: str, max_len: int):
    """
    Vectorized one-hot: returns numpy array shape (4, max_len), dtype float32.
    Unknown characters are left as zeros.
    """
    seq = seq[:max_len]  # truncate
    # bytes array of ASCII codes
    try:
        sb = seq.encode('ascii')  # fastest, will throw if non-ascii
    except Exception:
        sb = seq.encode('ascii', errors='ignore')
    if len(sb) == 0:
        return np.zeros((4, max_len), dtype=np.float32)
    codes = np.frombuffer(sb, dtype=np.uint8)
    idxs = _LU[codes]              # vectorized mapping to 0..4
    valid = idxs < 4               # boolean mask of A/C/G/T
    idxs = idxs[valid].astype(np.int64)
    cols = np.arange(len(idxs), dtype=np.int64)
    arr = np.zeros((4, max_len), dtype=np.float32)
    if idxs.size > 0:
        arr[idxs, cols] = 1.0
    return arr