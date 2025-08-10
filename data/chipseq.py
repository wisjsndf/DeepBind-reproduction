import pybedtools
from pyfaidx import Fasta
from torch.utils.data import Dataset
from utils import one_hot_encode

class ChIPSeqDataset(Dataset):
    def __init__(self, peaks_bed, fasta_file, flank=50):
        bed = pybedtools.BedTool(peaks_bed).slop(b=flank, genome='hg38')
        fasta = Fasta(fasta_file)
        raw = []
        for iv in bed:
            seq = fasta[iv.chrom][iv.start:iv.end].seq.upper()
            raw.append((seq, float(iv.score)))
        self.items = raw
        self.max_len = 2 * flank + 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, score = self.items[idx]
        mat = one_hot_encode(seq, max_len=self.max_len)
        return mat, score