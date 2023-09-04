import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i+1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

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


class Encoder(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mfcc=80,
        n_mels=128,
        spk_emb_dim=128,
        # gru_units=512,
        ):
        super().__init__()
        self.sr = sample_rate
        self.hop_length = hop_length
        self.speaker_encoder = SpeakerEncoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            spk_emb_dim=spk_emb_dim, 
        )
            
    def forward(self, signal, loundness, frequency):
        f0 = frequency.unsqueeze(dim=-1)
        # f0 = frequency
        z = self.speaker_encoder(signal)
        l = loundness.unsqueeze(dim=-1)
        # l = loundness
        return  (f0, z, l)


class Decoder(nn.Module):
    def __init__(
        self,
        mlp_layer=3,
        z_unit=16,
        gru_unit=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.mlp_f = mlp(1, gru_unit, mlp_layer) # input (batch, 250, 1), ouptut (batch, 250, 512)
        self.mlp_z = mlp(z_unit, gru_unit, mlp_layer) # input (batch, 250, 16), output (batch, 250, 512)
        self.mlp_l = mlp(1, gru_unit, mlp_layer) # input (batch, 250, 1), output (batch, 250, 512)
        self.gru = gru(3, gru_unit) # n_input=3 -> (f, z, l)
        self.mlp_gru = mlp(3*gru_unit, gru_unit, mlp_layer) # mlp input ?
        self.dense_harm = nn.Linear(gru_unit, n_harms + 1)
        self.dense_noise = nn.Linear(gru_unit, noise_filter_bank)
        
    def forward(self, encoder_output):
        # encoder_output -> (f0, z, l)
        out_mlp_f = self.mlp_f(encoder_output[0])
        out_mlp_z = self.mlp_z(encoder_output[1])
        out_mlp_l = self.mlp_l(encoder_output[2])
        out_cat_mlps = torch.cat((out_mlp_f, out_mlp_z, out_mlp_l), dim=2)
        out_gru, _ = self.gru(out_cat_mlps)
        out_cat_gru_mlps = torch.cat((out_gru, out_mlp_f, out_mlp_l), dim=2)
        out_mlp_gru = self.mlp_gru(out_cat_gru_mlps)
        
        # harmonic part
        out_dense_harm = self.dense_harm(out_mlp_gru)
        # out_dense_harmonic output -> 1(global_amplitude) + n_harmonics 
        global_amp = self.modified_sigmoid(out_dense_harm[..., :1])
        n_harm_amps = out_dense_harm[..., 1:]
        # n_harm_amps /= n_harm_amps.sum(-1, keepdim=True) # not every element >= 0
        n_harm_amps_norm = F.softmax(n_harm_amps, dim=-1)
        harm_amp_distribution = global_amp * n_harm_amps_norm

        # noise filter part
        out_dense_noise = self.dense_noise(out_mlp_gru)
        noise_filter_bank = self.modified_sigmoid(out_dense_noise)

        return harm_amp_distribution, noise_filter_bank

    # force the amplitudes, harmonic distributions, and filtered noise magnitudes 
    # to be non-negative by applying a sigmoid nonlinearity to network outputs.
    @staticmethod
    def modified_sigmoid(x):
        return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7
