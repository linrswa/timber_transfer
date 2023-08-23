#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import os
import numpy as np
from glob import glob

class NSynthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_mode: str,
        dir_path: str = "/dataset/NSynth/nsynth-subset",
        sr: int = 16000,
        frequency_with_confidence: bool = False,
    ):
        super().__init__()
        self.sr = sr
        self.dir_path = f"{dir_path}/{data_mode}"
        signal_path = os.path.join(self.dir_path, "signal/*")
        self.audio_list = glob(signal_path)
        self.info_type = (
            ("signal", "loudness", "frequency_c")
            if frequency_with_confidence
            else ("signal", "loudness", "frequency")
        )

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        signal_path = self.audio_list[idx]
        file_name = signal_path.split("/")[-1][:-4]
        signal = np.load(
            os.path.join(self.dir_path, f"{self.info_type[0]}/{file_name}.npy")
        ).astype("float32")
        loudness = np.load(
            os.path.join(self.dir_path, f"{self.info_type[1]}/{file_name}.npy")
        ).astype("float32")[..., :-1]
        frequency = np.load(
            os.path.join(self.dir_path, f"{self.info_type[2]}/{file_name}.npy")
        ).astype("float32")

        return (file_name, signal, loudness, frequency)

# from nvc-net paper
class Speaker_encoder(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        n_mfcc=80,
        spk_emb_dim=128, 
        ):

        super().__init__()
        self.extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=dict(
                n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=20.0, f_max=8000.0
            )
        )
        self.conv = nn.Sequential(
            nn.Conv1d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.downblocks_list = nn.ModuleList(
            [self.build_downblock(32 * 2 ** i) for i in range(4) ]
        )

        self.downblocks = nn.Sequential(*self.downblocks_list)

        self.conv_mean = nn.Conv1d(512, spk_emb_dim, kernel_size=1, stride=1, padding=0)

        self.conv_convariance = nn.Conv1d(512, spk_emb_dim, kernel_size=1, stride=1, padding=0)

    def build_downblock(self, in_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )
    
    def forward(self, x):
        x = self.extract_mfcc(x)
        x = self.conv(x)
        x = self.downblocks(x)
        x = nn.AvgPool1d(x.size(-1))(x)
        mean_emb = self.conv_mean(x)
        convariance_emb = self.conv_convariance(x)
        return mean_emb, convariance_emb
        
from torch.utils.data import DataLoader

train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

model = Speaker_encoder()