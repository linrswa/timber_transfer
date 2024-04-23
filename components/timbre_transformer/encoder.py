import torch.nn as nn
import torchaudio


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
    

class MultiDimEmbHeader(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_dense1 = nn.Linear(128, 256)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, timbre_emb):
        dd1 = self.relu(self.down_dense1(timbre_emb))
        return timbre_emb, dd1
        

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, loundness, f0):
        f0 = f0.unsqueeze(dim=-1)
        l = loundness.unsqueeze(dim=-1)
        return  f0, l