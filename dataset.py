import utils

import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from natsort import natsorted
import numpy as np

from os import listdir
from os.path import join


def sin_wave_data(batch_size, timesteps):
    low_freq_factor = np.random.uniform(size=(batch_size,))
    high_freq_factor = np.random.uniform(size=(batch_size,))

    x = np.arange(0, np.pi, np.pi/timesteps)
    low_y = (low_freq_factor + 1)[:, None]*x[None, :]
    high_y = 20.*(high_freq_factor + 1)[:, None]*x[None, :]

    noise = np.random.uniform(low=-0.001, high=0.001, size=high_y.shape)
    batch = np.sin(high_y)*np.sin(low_y) + noise
    return batch.astype('float32')


class FolderDataset(Dataset):

    toy_data_count = 1024
    toy_seq_len = 16000 * 8

    def __init__(self, path=None, overlap_len=64, q_levels=0,
                    ratio_min=0, ratio_max=1, toy_sin_wave = False):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        self.toy_sin_wave = toy_sin_wave
        if(toy_sin_wave == False):
            file_names = natsorted(
                [join(path, file_name) for file_name in listdir(path)]
            )
            self.file_names = file_names[
                int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))
            ]

    def __getitem__(self, index):
        if(self.toy_sin_wave == False):
            (seq, _) = load(self.file_names[index], sr=None, mono=True)
            # print(self.file_names[index])
            return torch.cat([
                torch.zeros(self.overlap_len),
                torch.from_numpy(seq)
            ])
        else:
            return torch.from_numpy(
                sin_wave_data(1, self.toy_seq_len + self.overlap_len).reshape(-1,)
                )

    def __len__(self):
        if(self.toy_sin_wave == True):
            return self.toy_data_count
        else:
            return len(self.file_names)


class DataLoader(DataLoaderBase):

    def __init__(self, dataset, batch_size, seq_len, overlap_len,
                 *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.seq_len = seq_len
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            (batch_size, n_samples) = batch.size()

            reset = True

            for seq_begin in range(self.overlap_len, n_samples, self.seq_len):
                from_index = seq_begin - self.overlap_len
                to_index = seq_begin + self.seq_len
                sequences = batch[:, from_index : to_index]
                input_sequences = sequences[:, : -1]
                target_sequences = sequences[:, self.overlap_len :].contiguous()

                # yield (input_sequences, reset, target_sequences)
                yield (sequences, reset, target_sequences)

                reset = False

    def __len__(self):
        raise NotImplementedError()
