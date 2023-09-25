import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


# from nvc-net paper
class TimbreEncoder(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        n_mfcc=80,
        timbre_emb_dim=256 
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
        self.down_dense1 = nn.Linear(256, 128)
        self.down_dense2 = nn.Linear(128, 64)

    def forward(self, timbre_emb):
        dd1 = self.down_dense1(timbre_emb)
        dd2 = self.down_dense2(dd1)
        return dd2, dd1, timbre_emb
        

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, loundness, f0):
        f0 = f0.unsqueeze(dim=-1)

        # timbre_emb = self.timbre_encoder(signal)
        # multi_dim_emb = self.multi_dim_emb_header(timbre_emb)

        l = loundness.unsqueeze(dim=-1)
        return  f0, l


class TCUB(nn.Module):
    def __init__(self, temporal, in_ch, out_ch):
        super().__init__()
        self.conv_1x1_input = nn.Conv1d(temporal, temporal, kernel_size=1, stride=1, padding=0)
        self.conv_1x1_condition = nn.Conv1d(temporal, temporal, kernel_size=1, stride=1, padding=0)
        self.dense_upsample = nn.Linear(in_ch, out_ch)
        self.conv_1x1_output = nn.Conv1d(temporal, temporal, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x, condition):
        x_input = self.conv_1x1_input(x)
        x_condition = self.conv_1x1_condition(condition)
        x_upsample = self.dense_upsample(x)
        mix = torch.cat([x_input, x_condition], dim=-1)
        mix_tanh = torch.tanh(mix)
        mix_sigmoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_sigmoid
        mix_output = self.conv_1x1_output(mix_output)
        output = nn.LeakyReLU(0.2)(x_upsample + mix_output) 
        return output 


class Decoder(nn.Module):
    def __init__(
        self,
        mlp_layer=3,
        temporal = 250,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.mlp_f0 = self.mlp(1, 32, mlp_layer)
        self.mlp_loudness = self.mlp(1, 32, mlp_layer)

        self.tcrb_1 = TCUB(temporal=temporal, in_ch=64, out_ch=128)
        self.tcrb_2 = TCUB(temporal=temporal, in_ch=128, out_ch=256)
        self.tcrb_3 = TCUB(temporal=temporal, in_ch=256, out_ch=512)
    
        self.mlp_final = self.mlp(512 + 64, 512, mlp_layer) # mlp input ?
        self.dense_harm = nn.Linear(512, n_harms + 1)
        self.dense_noise = nn.Linear(512, noise_filter_bank)
        
    def forward(self, f0, loudness, multi_timbre_emb):
        # encoder_output -> (f0, (timbre_emb64, timbre_emb128, timbre_emb256), l)
        out_mlp_f0 = self.mlp_f0(f0)
        out_mlp_loudness = self.mlp_loudness(loudness)
        out_cat_mlp = torch.cat([out_mlp_f0, out_mlp_loudness], dim=-1)
        out_tcrb_1 = self.tcrb_1(out_cat_mlp, multi_timbre_emb[0].expand_as(out_cat_mlp).contiguous())
        out_tcrb_2 = self.tcrb_2(out_tcrb_1, multi_timbre_emb[1].expand_as(out_tcrb_1).contiguous())
        out_tcrb_3 = self.tcrb_3(out_tcrb_2, multi_timbre_emb[2].expand_as(out_tcrb_2).contiguous())
        out_cat_f0_loudness = torch.cat([out_tcrb_3, out_mlp_f0, out_mlp_loudness], dim=-1)
        out_mlp_final = self.mlp_final(out_cat_f0_loudness)
        
        # harmonic part
        out_dense_harm = self.dense_harm(out_mlp_final)
        # out_dense_harmonic output -> 1(global_amplitude) + n_harmonics 
        global_amp = self.modified_sigmoid(out_dense_harm[..., :1])
        n_harm_amps = out_dense_harm[..., 1:]
        # n_harm_amps /= n_harm_amps.sum(-1, keepdim=True) # not every element >= 0
        n_harm_amps_norm = F.softmax(n_harm_amps, dim=-1)
        harm_amp_distribution = global_amp * n_harm_amps_norm

        # noise filter part
        out_dense_noise = self.dense_noise(out_mlp_final)
        noise_filter_bank = self.modified_sigmoid(out_dense_noise)

        return harm_amp_distribution, noise_filter_bank

    # force the amplitudes, harmonic distributions, and filtered noise magnitudes 
    # to be non-negative by applying a sigmoid nonlinearity to network outputs.
    @staticmethod
    def modified_sigmoid(x):
        return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

    @staticmethod
    def mlp(in_size, hidden_size, n_layers):
        channels = [in_size] + (n_layers) * [hidden_size]
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(channels[i], channels[i + 1]))
            net.append(nn.LayerNorm(channels[i+1]))
            net.append(nn.LeakyReLU())
        return nn.Sequential(*net)