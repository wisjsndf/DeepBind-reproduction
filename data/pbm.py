import pandas as pd
from torch.utils.data import Dataset
from utils import one_hot_encode

class PBMDataset(Dataset):
    def __init__(self, seq_file, target_file, tf_col, max_len=36):
        df_seq = pd.read_csv(seq_file, sep='\t')
        df_target = pd.read_csv(target_file, sep='\t')
        raw = [
            (s.upper(), float(y)) for s, y in
            zip(df_seq['seq'], df_target[tf_col])
            if pd.notna(y)
        ]
        self.items = raw
        self.max_len = max_len
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        seq, label = self.items[idx]
        x = one_hot_encode(seq, self.max_len)
        return x, label