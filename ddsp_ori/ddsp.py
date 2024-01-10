import torch.nn as nn
from .autoencoder import Encoder, Decoder
from .component import HarmonicOscillator, NoiseFilter, Reverb

class DDSP(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mfcc=30,
        n_mels=128,
        gru_units=512,
        z_units=16,
        mlp_layer=3,
        n_harms=101,
        noise_filter_bank=65, 
        ):
        super().__init__()
        self.encoder = Encoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            gru_units=gru_units,
            z_units=z_units, 
        )
        self.decoder = Decoder(
            mlp_layer=mlp_layer,
            z_unit=z_units,
            gru_unit=gru_units,
            n_harms=n_harms,
            noise_filter_bank=noise_filter_bank
        )
        self.synthesizer = HarmonicOscillator(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_harms=n_harms,
        )
        self.noise_filter = NoiseFilter(
            hop_length=hop_length
        )
    
    def forward(self, signal, loudness, frequency):
        f0, z, l = self.encoder(signal, loudness, frequency)

        harm_amp_distribution, noise_filter_bank = self.decoder((f0, z, l))

        additive_output = self.synthesizer(harm_amp_distribution, f0)

        subtractive_output = self.noise_filter(noise_filter_bank)

        reconstruct_signal = additive_output + subtractive_output
        # reconstruct_signal = additive_output

        return additive_output, subtractive_output, reconstruct_signal