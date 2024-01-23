import torch
import math
import torch.nn as nn
from .utils_blocks import UpFusionBlock, DFBlock

# force the amplitudes, harmonic distributions, and filtered noise magnitudes 
# to be non-negative by applying a sigmoid nonlinearity to network outputs.
def modified_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7): 
    return max_value * torch.sigmoid(x)**math.log(exponent) + threshold

class AmpStack(nn.Module):
    def __init__(self, emb_dim=8):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_dim, 1),
            )
    
    def forward(self, amp):
        amp_att = self.stack(amp)
        return modified_sigmoid(amp + amp_att)


class HarmonicHead(nn.Module):
    def __init__(self, in_size, timbre_emb_size, n_harms):
        super().__init__()
        self.dense_harm = nn.Linear(in_size, n_harms+1)
        self.dfblock1 = DFBlock(n_harms, timbre_emb_size, affine_dim=n_harms, out_layer_mlp=True)
        self.dfblock2 = DFBlock(n_harms, timbre_emb_size, affine_dim=n_harms, out_layer_mlp=True)
        self.relu = nn.LeakyReLU(0.2)
        self.stack_amp = AmpStack(emb_dim=8)

    def forward(self, out_mlp_final, timbre_emb):
        n_harm_amps = self.dense_harm(out_mlp_final)

        # out_dense_harmonic output -> global_amplitude(1) + n_harmonics(101) 
        global_amp, n_harm_dis = n_harm_amps[..., :1], n_harm_amps[..., 1:]

        # harmonic distribution part
        n_harm_dis = self.relu(n_harm_dis)
        df_out = self.dfblock1(n_harm_dis, timbre_emb)
        df_out = self.dfblock2(df_out, timbre_emb)
        n_harm_dis = n_harm_dis + df_out

        # global amplitude part
        global_amp = self.relu(global_amp)
        global_amp = self.stack_amp(global_amp)

        # n_harm_amps /= n_harm_amps.sum(-1, keepdim=True) # not every element >= 0
        n_harm_dis_norm = nn.functional.softmax(n_harm_dis, dim=-1)

        return n_harm_dis_norm, global_amp


class NoiseHead(nn.Module):
    def __init__(self, in_size, noise_filter_bank):
        super().__init__()
        self.dense_noise = nn.Linear(in_size, noise_filter_bank + 1)
        self.relu = nn.LeakyReLU(0.2)
        self.stack_amp = AmpStack(emb_dim=8)
    
    def forward(self, out_mlp_final):
        out_dense_noise = self.dense_noise(out_mlp_final)
        global_amp, noise_filter_bank = out_dense_noise[..., :1], out_dense_noise[..., 1:]
        noise_filter_bank = self.relu(noise_filter_bank)
        global_amp = self.stack_amp(global_amp)

        return noise_filter_bank, global_amp


class Decoder(nn.Module):
    def __init__(
        self,
        in_extract_size=64,
        mlp_layer=3,
        timbre_emb_size=128,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.mlp_f0 = self.mlp(1, in_extract_size, mlp_layer)
        self.mlp_loudness = self.mlp(1, in_extract_size, mlp_layer)
        self.gru_loudness = nn.GRU(in_extract_size, in_extract_size, num_layers=3, batch_first=True)
        self.gru_f0 = nn.GRU(in_extract_size, in_extract_size, num_layers=3, batch_first=True)
        in_size = in_extract_size * 2

        self.gru_timbre_1 = nn.GRU(timbre_emb_size, timbre_emb_size, batch_first=True)
        self.gru_timbre_2 = nn.GRU(timbre_emb_size, timbre_emb_size, batch_first=True)

        self.upfusionblock_1 = UpFusionBlock(in_ch=in_size, emb_dim=timbre_emb_size) # in_size = 128
        self.upfusionblock_2 = UpFusionBlock(in_ch=in_size*2, emb_dim=timbre_emb_size) # in_size = 256
    
        self.mlp_final = self.mlp(in_size * 4 + in_size, final_embedding_size, mlp_layer) 
        self.harmonic_head = HarmonicHead(final_embedding_size, timbre_emb_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        out_mlp_f0 = self.mlp_f0(f0)
        out_mlp_loudness = self.mlp_loudness(loudness)
        out_cat_mlp = torch.cat([out_mlp_f0, out_mlp_loudness], dim=-1)

        out_before_up = out_cat_mlp.transpose(1, 2).contiguous()
        timbre_emb_1, hidden_state_1 = self.gru_timbre_1(timbre_emb)
        out_upfb_1 = self.upfusionblock_1(out_before_up, timbre_emb_1)
        timbre_emb_2, _ = self.gru_timbre_2(timbre_emb_1, hidden_state_1)
        out_upfb_2 = self.upfusionblock_2(out_upfb_1, timbre_emb_2)
        out_after_up = out_upfb_2.transpose(1, 2).contiguous()

        out_gru_loudness, _ = self.gru_loudness(out_mlp_loudness)
        out_gru_f0, _ = self.gru_f0(out_mlp_f0)
        out_cat_f0_loudness = torch.cat([out_after_up, out_gru_f0, out_gru_loudness], dim=-1)
        out_mlp_final = self.mlp_final(out_cat_f0_loudness)
        
        # harmonic part
        harmonic_output = self.harmonic_head(out_mlp_final, timbre_emb)

        # noise filter part
        noise_output = self.noise_head(out_mlp_final)

        return harmonic_output, noise_output


    @staticmethod
    def mlp(in_size, hidden_size, n_layers):
        channels = [in_size] + (n_layers) * [hidden_size]
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(channels[i], channels[i + 1]))
            net.append(nn.LayerNorm(channels[i+1]))
            net.append(nn.LeakyReLU())
        return nn.Sequential(*net)

