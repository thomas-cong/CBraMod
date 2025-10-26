import pickle

import lmdb
import torch
import torchaudio
from torch.utils.data import Dataset

from utils.util import to_tensor


class PretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dir,
            n_fft=398,
            use_spectrogram=False
    ):
        super(PretrainingDataset, self).__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        # self.keys = self.keys[:100000]
        
        self.use_spectrogram = use_spectrogram
        self.n_fft = n_fft
        if use_spectrogram:
            # Initialize spectrogram transform (will be moved to device in __getitem__)
            self.spectrogram_transform = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))

        patch = to_tensor(patch)  # Shape: (19, 30, 200)
        if self.use_spectrogram:
            if self.spectrogram_transform is None:
                hop_length = int(patch.shape[-1] * 0.02)  # 2% of sequence length
                self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=hop_length).to(patch.device)
            patch = make_spectrogram(patch, self.spectrogram_transform)
        return patch
if __name__ == "__main__":
    dataset = PretrainingDataset('/orcd/scratch/bcs/001/fiete/bfm_dataset/preprocessed/tueg/tuh_eeg_0.lmdb', use_spectrogram=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)  # Use 0 for debugging
    for x in dataloader:
        print(x.shape)
        break