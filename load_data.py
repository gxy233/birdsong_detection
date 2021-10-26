import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display as display
import librosa
import IPython.display as ipd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import random_split, DataLoader, Dataset
import torchaudio.transforms as T
import torchvision
from tqdm.notebook import tqdm
import random

batch_size = 128
class BirdVox70kDS(Dataset):
        def __init__(self, root_dir, fnames, transforms=None):
            # store transforms func
            self.transforms = transforms
            # initialize storage arrays
            self.wave_loc = []
            self.labels = []

            # for each hdf5 file...
            for fname in fnames:
                # open the file
                fhdf5 = os.path.join(root_dir, fname)
                with h5py.File(fhdf5, 'r') as f:
                    # navigate to `waveforms` group
                    waveforms = f['waveforms']
                    # for each piece of data...
                    for waveform in waveforms.keys():
                        # append waveform filename for later access
                        self.wave_loc.append([fhdf5, waveform])
                        # (label == last digit of filename)
                        self.labels.append(waveform[-1])

            # turn them into np.arrays
            self.wave_loc = np.array(self.wave_loc)
            self.labels = np.array(self.labels)

            # melspec transform (similar to `librosa.feature.melspectrogram()`)
            self.melspec = T.MelSpectrogram(sample_rate=24000,
                                            n_fft=2048,
                                            hop_length=512)

        def __len__(self):
            # size of dataset
            return len(self.labels)

        def __getitem__(self, idx):
            # fetch waveform from hdf5 file & label
            fhdf5, waveform = self.wave_loc[idx]
            with h5py.File(fhdf5, 'r') as f:
                wave = f['waveforms'][waveform]
                # convert to np array for faster i/o performance
                # ^^ https://github.com/pytorch/pytorch/issues/28761
                wave = np.array(wave)
                # apply other specified transforms
                if self.transforms:
                    wave = self.transforms()(wave)
                # convert into tensor & apply melspec
                wave = self.melspec(torch.Tensor(wave))
                # unsqueeze adds dimension needed for pytorch's `Conv2d`
                wave = wave.unsqueeze(0)
            # parse label (still a string)
            label = self.labels[idx]
            return wave, int(label)

root_dir = r'D:\BirdVox-70k'
fnames = ['BirdVox-70k_unit01.hdf5',
          'BirdVox-70k_unit02.hdf5',
          'BirdVox-70k_unit03.hdf5']


train_ds = BirdVox70kDS(root_dir, fnames)
val_ds = BirdVox70kDS(root_dir, ['BirdVox-70k_unit05.hdf5'])
train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size * 2, pin_memory=True)