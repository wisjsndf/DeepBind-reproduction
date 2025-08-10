import pandas as pd
from torch.utils.data import Dataset
from utils import one_hot_encode

class RNAcompeteDataset(Dataset):
    def __init__(self, raw_file, max_len=41):
        df = pd.read_csv(raw_file, sep='\t')
        intensity_cols = [c for c in df.columns if 'Intensity' in c]
        raw = [
            (s.upper(), df.loc[i, intensity_cols].mean())
            for i, s in enumerate(df['ProbeSequence'])
        ]
        self.items = raw
        self.max_len = max_len

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        seq, label = self.items[idx]
        x = one_hot_encode(seq, self.max_len)
        return x, label
    