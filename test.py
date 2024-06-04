#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
from data.dataset import NSynthDataset

train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 
class ZEncoder(nn.Module):
    def __init__(self, nfft=1024, hop_lenght=256, z_units=16, hidden_size=256):
        super().__init__()
        self.nfft = nfft
        self.hop_lenght = hop_lenght
        input_size = nfft // 2 + 1
        self.norm = nn.InstanceNorm1d(input_size, affine=True)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, z_units)
    
    def forward(self, x):
        x = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.hop_lenght,
            win_length=self.nfft,
            center=True,
            return_complex=True,
        )
        x = x[..., :-1]
        x = torch.abs(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous() # (batch, nfft, frame) -> (batch, frame, nfft)
        x = self.gru(x)[0]
        x = self.dense(x)
        return x

zencoder = ZEncoder()
s = zencoder(s)