import torch
import torch.nn as nn
import torchaudio
from functools import partial


# from nvc-net paper
class TimbreEncoder(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        n_mfcc=80,
        timbre_emb_dim=256,
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
            nn.LeakyReLU(0.2),
        )

        self.downblocks_list = nn.ModuleList(
            [self.build_downblock(32 * 2 ** i) for i in range(4) ]
        )

        self.downblocks = nn.Sequential(*self.downblocks_list)

        self.conv_mean = nn.Conv1d(512, timbre_emb_dim, kernel_size=1, stride=1, padding=0)

        self.conv_covariance = nn.Conv1d(512, timbre_emb_dim, kernel_size=1, stride=1, padding=0)

    def build_downblock(self, in_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.AvgPool1d(2),
        )
    
    def forward(self, x):
        x = self.extract_mfcc(x)
        x = self.conv(x)
        x = self.downblocks(x)
        x = nn.AvgPool1d(x.size(-1))(x)
        mu = self.conv_mean(x)
        logvar = self.conv_covariance(x)

        return mu, logvar
    
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


class ZMFCCEncoder(nn.Module):
    def __init__(
        self,
        nfft=1024,
        hop_lenght=256,
        z_units=16,
        gru_units=256,
        n_mfcc=30,
        n_mels=128,
        ):
        super().__init__()
        self.extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=nfft, hop_length=hop_lenght, n_mels=n_mels, f_min=20.0, f_max=8000.0
            )
        )
            
        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            batch_first=True,
        )
        self.dense = nn.Linear(gru_units, z_units)
    
    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.extract_mfcc(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous() # (batch, nfft, frame) -> (batch, frame, nfft)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x[..., :-1, :]

class EngryEncoder(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.extract_fn = partial(self.extract_frames, n_fft=n_fft, hop_length=hop_length)

    def forward(self, signal):
        frames = self.extract_fn(signal)
        if frames.dim() == 2:
            frames = frames.unsqueeze(dim=0)
        frames = frames[:, :-1, :].contiguous()
        energy = torch.sqrt((frames**2).mean(dim=-1, keepdim=True))
        return energy

    @staticmethod
    def extract_frames(signal, n_fft, hop_length, center=True):
        if center:
            signal = torch.nn.functional.pad(signal, (n_fft // 2, n_fft // 2), mode='reflect')
        
        num_frames = (signal.shape[-1] - n_fft) // hop_length + 1
        indices = torch.arange(0, num_frames * hop_length, hop_length).unsqueeze(1) + torch.arange(n_fft).unsqueeze(0)
        frames = signal[:, indices].squeeze(0)
        return frames


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.zencoder = ZMFCCEncoder()
        self.engry_encoder = EngryEncoder()
            
    def forward(self, signal, loundness, f0):
        f0 = f0.unsqueeze(dim=-1)
        l = loundness.unsqueeze(dim=-1)
        engry = self.engry_encoder(signal)
        # z = self.zencoder(signal)
        return  f0, l, engry