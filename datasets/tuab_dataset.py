import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
from utils.signaltools import make_spectrogram
import torchaudio
import os
import random
import lmdb
import pickle
from scipy import signal

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            use_spectrogram = False
    ):
        super(CustomDataset, self).__init__()
        self.files = [os.path.join(data_dir, mode, file) for file in os.listdir(os.path.join(data_dir, mode))]
        self.use_spectrogram = use_spectrogram
        if use_spectrogram:
            # Initialize spectrogram transform on CPU
            hop_length = 40 # 2% of 2000 (sequence length after reshape)
            self.spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=398, hop_length=hop_length)
    def __len__(self):
        return len((self.files))

    def __getitem__(self, idx):
        file = self.files[idx]
        data_dict = pickle.load(open(file, 'rb'))
        data = data_dict['X']
        label = data_dict['y']
        # data = signal.resample(data, 2000, axis=-1)
        data = data.reshape(16, 10, 200)
        if self.use_spectrogram:
            # Convert to tensor first, then apply spectrogram
            data = torch.from_numpy(data).float()
            data = make_spectrogram(data, self.spectrogram_transform)
            return data.numpy(), label
        return data/100, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train', use_spectrogram=self.params.use_spectrogram)
        val_set = CustomDataset(self.datasets_dir, mode='val', use_spectrogram=self.params.use_spectrogram)
        test_set = CustomDataset(self.datasets_dir, mode='test', use_spectrogram=self.params.use_spectrogram)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set) + len(val_set) + len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader
if __name__ == "__main__":
    data_dir = "/orcd/scratch/bcs/001/fiete/bfm_dataset/preprocessed/tuab"
    train_files = os.listdir(os.path.join(data_dir, "train"))
    dataset = CustomDataset(os.path.join(data_dir, "train"), train_files, use_spectrogram=True)
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate, num_workers=0)  # Use 0 for debugging
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
