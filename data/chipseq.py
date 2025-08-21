import pybedtools
from pyfaidx import Fasta
from torch.utils.data import Dataset
from utils import one_hot_encode
import pandas as pd
import torch

try:
    import pybedtools
    from pyfaidx import Fasta
    _HAS_PYBEDTOOLS = True
except Exception:
    _HAS_PYBEDTOOLS = False

class ChIPSeqDataset(Dataset):
    def __init__(
        self,
        seq_file: str = None,
        peaks_bed: str = None,
        fasta_file: str = None,
        max_len: int = 101,
        flank: int = 50,
        sep: str = '\t',
        binarize: bool = False,
        bin_thresh: float = 0.0
    ):
        self.items = []
        self.max_len = max_len
        if seq_file is not None:
            df = pd.read_csv(seq_file, sep=sep, header=None, comment='#', engine='python')
            if df.shape[1] < 2:
                raise ValueError("seq_file must have at least 2 columns: sequence and label")
            seqs = df.iloc[:, 0].astype(str).str.upper().tolist()
            labels = df.iloc[:, -1].tolist()
            for s, y in zip(seqs, labels):
                if pd.isna(y):
                    continue
                if binarize:
                    y = 1.0 if float(y) > bin_thresh else 0.0
                self.items.append((s, float(y)))
        else:
            if not _HAS_PYBEDTOOLS:
                raise ImportError("pybedtools and pyfaidx are required for BED+FASTA mode")
            if peaks_bed is None or fasta_file is None:
                raise ValueError("Either seq_file or (peaks_bed and fasta_file) must be provided")
            bed = pybedtools.BedTool(peaks_bed)
            try:
                bed_slopped = bed.slop(b=flank, genome='hg38')
            except Exception:
                bed_slopped = bed
            fasta = Fasta(fasta_file)
            for iv in bed_slopped:
                try:
                    seq = fasta[iv.chrom][int(iv.start):int(iv.end)].seq.upper()
                except Exception:
                    continue
                try:
                    score = float(iv.score)
                except Exception:
                    score = 1.0
                if binarize:
                    score = 1.0 if score > bin_thresh else 0.0
                self.items.append((seq, float(score)))
        if len(self.items) == 0:
            raise ValueError("No sequences loaded into ChIPSeqDataset")
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        seq, label = self.items[idx]
        x = one_hot_encode(seq, self.max_len)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y