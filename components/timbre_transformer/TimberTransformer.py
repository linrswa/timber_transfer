import torch
import torch.nn as nn

from .encoders import Encoder, TimbreEncoder
from .decoders import  Decoder
from .component import HarmonicOscillator, NoiseFilter

class TimbreTransformer(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mfcc=80,
        n_mels=128,
        timbre_emb_dim=128,
        n_harms=101,
        noise_filter_bank=65, 
        is_train=False,
        is_smooth=False,
        ):

        super().__init__()

        self.is_train = is_train

        self.timbre_encoder = TimbreEncoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            timbre_emb_dim=timbre_emb_dim, 
        )

        self.encoder = Encoder()

        self.decoder = Decoder(
            n_harms=n_harms,
            noise_filter_bank=noise_filter_bank,
            timbre_emb_size=timbre_emb_dim
        )

        self.synthesizer = HarmonicOscillator(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_harms=n_harms,
            is_smooth=is_smooth,
        )

        self.noise_filter = NoiseFilter(
            hop_length=hop_length
        )
    
    def forward(self, signal, loudness, f0):
        
        f0, l = self.encoder(loudness, f0)
        mu, logvar = self.timbre_encoder(signal)
        timbre_emb = self.sample(mu, logvar)

        harmonic_head_output, noise_head_output = self.decoder(f0, l, timbre_emb)

        additive_output = self.synthesizer(harmonic_head_output, f0)

        subtractive_output = self.noise_filter(noise_head_output)

        reconstruct_signal = additive_output + subtractive_output

        global_amp = harmonic_head_output[1]

        return additive_output, subtractive_output, reconstruct_signal, mu, logvar, global_amp

    def sample(self, mu, logvar):
        """ paper discription
        Thus, a speaker embedding is given by sampling from the output distribution, i.e., z ∼ N (µ,σ^2I). 
        Although the sampling operation is non-differentiable, it can be reparameterized as a differentiable 
        operation using the reparameterization trick [26], i.e., z = µ + σ (.) epsilon, where epsilon ∼ N (0, I).
        """
        # Assuming mean_emb and covariance_emb are obtained from the speaker encoder
        # mean_emb and covariance_emb should be PyTorch tensors

        if self.is_train:
            # Sample epsilon from a normal distribution with mean 0 and standard deviation 1
            epsilon = torch.randn_like(logvar)

            # Reparameterize the speaker embedding
            timbre_emb = mu + torch.exp(0.5 * logvar) * epsilon 
            # the line above ref https://github.com/sony/ai-research-code/blob/master/nvcnet/model/model.py#L16
            timbre_emb = timbre_emb.permute(0, 2, 1).contiguous() # (batch, spk_emb_dim, 1) -> (batch, 1, spk_emb_dim)
            return timbre_emb

        mu = mu.permute(0, 2, 1).contiguous() # (batch, spk_emb_dim, 1) -> (batch, 1, spk_emb_dim)
        return mu