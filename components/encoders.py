#%%
import torch
import torch.nn as nn
import torchaudio


# from nvc-net paper
class SpeakerEncoder(nn.Module):
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

        self.conv_covariance = nn.Conv1d(512, spk_emb_dim, kernel_size=1, stride=1, padding=0)

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
        covariance_emb = self.conv_covariance(x)

        """ paper discription
        Thus, a speaker embedding is given by sampling from the output distribution, i.e., z ∼ N (µ,σ^2I). 
        Although the sampling operation is non-differentiable, it can be reparameterized as a differentiable 
        operation using the reparameterization trick [26], i.e., z = µ + σ (.) epsilon, where epsilon ∼ N (0, I).
        """
        # Assuming mean_emb and covariance_emb are obtained from the speaker encoder
        # mean_emb and covariance_emb should be PyTorch tensors

        # Sample epsilon from a normal distribution with mean 0 and standard deviation 1
        epsilon = torch.randn_like(covariance_emb)

        # Reparameterize the speaker embedding
        speaker_embedding = mean_emb + torch.sqrt(covariance_emb) * epsilon
        return speaker_embedding
        