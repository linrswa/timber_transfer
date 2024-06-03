#%%
import torch
import math
import torch.nn as nn
from ..utils_blocks import AttSubBlock

from ..utils import safe_divide

# force the amplitudes, harmonic distributions, and filtered noise magnitudes 
# to be non-negative by applying a sigmoid nonlinearity to network outputs.
def modified_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7): 
    return max_value * torch.sigmoid(x)**math.log(exponent) + threshold

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            self.linear_stack(in_size, hidden_size),
            self.linear_stack(hidden_size, hidden_size),
            self.linear_stack(hidden_size, hidden_size),
        )
    
    def forward(self, x):
        return self.net(x)

    def linear_stack(self, in_size, hidden_size):
        block = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU()
            )
        return block

class InputAttBlock(nn.Module):
    def __init__(self, in_size=1, hidden_size=512, out_size=128):
        super().__init__()
        if hidden_size % 8 == 0:
            num_heads = 8
        else:
            num_heads = 1
        self.mlp = MLP(in_size, hidden_size)
        self.gru = nn.GRU(hidden_size, out_size, num_layers=3, batch_first=True)
        self.self_att = AttSubBlock(out_size, num_heads)
    
    def forward(self, x):
        x = self.mlp(x)
        x, _ = self.gru(x)
        x = self.self_att(x, x)
        return x


class HarmonicHead(nn.Module):
    def __init__(self, in_size, timbre_emb_size, n_harms):
        super().__init__()
        self.dense_harm = nn.Linear(in_size, n_harms+1)

    def forward(self, out_mlp_final):
        n_harm_amps = self.dense_harm(out_mlp_final)

        # out_dense_harmonic output -> global_amplitude(1) + n_harmonics(101) 
        n_harm_amps = modified_sigmoid(n_harm_amps)
        global_amp, n_harm_dis = n_harm_amps[..., :1], n_harm_amps[..., 1:]

        n_harm_dis_norm =  safe_divide(n_harm_dis, n_harm_dis.sum(dim=-1, keepdim=True)) 

        return n_harm_dis_norm, global_amp


class NoiseHead(nn.Module):
    def __init__(self, in_size, noise_filter_bank):
        super().__init__()
        self.dense_noise = nn.Linear(in_size, noise_filter_bank)
    
    def forward(self, out_mlp_final):
        out_dense_noise = self.dense_noise(out_mlp_final)
        noise_filter_bank = modified_sigmoid(out_dense_noise)

        return noise_filter_bank
    

class Decoder(nn.Module):
    def __init__(
        self,
        in_extract_size=512,
        timbre_emb_size=256,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.f0_mlp = MLP(1, in_extract_size)
        self.l_mlp = MLP(1, in_extract_size)
        self.f0_gru = nn.GRU(in_extract_size, timbre_emb_size, batch_first=True)
        self.l_gru = nn.GRU(in_extract_size, timbre_emb_size, batch_first=True)
        self.f0_self_att = AttSubBlock(timbre_emb_size, 8)
        self.l_self_att = AttSubBlock(timbre_emb_size, 8)

        self.mix_gru = nn.GRU(timbre_emb_size * 3, timbre_emb_size, batch_first=True)

        final_size = timbre_emb_size + in_extract_size * 2
        self.final_mlp = MLP(final_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, timbre_emb_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        out_f0_mlp = self.f0_mlp(f0)
        out_l_mlp = self.l_mlp(loudness)
        timbre_P = timbre_emb.permute(1, 0, 2).contiguous()
        out_f0_gru, _ = self.f0_gru(out_f0_mlp, timbre_P)
        out_l_gru, _ = self.l_gru(out_l_mlp, timbre_P)
        out_f0_self_att = self.f0_self_att(out_f0_gru, out_f0_gru)
        out_l_self_att = self.l_self_att(out_l_gru, out_l_gru)
        cat_input = torch.cat([out_f0_self_att, out_l_self_att, timbre_emb.expand_as(out_f0_gru)], dim=-1)
        out_mix_gru, _ = self.mix_gru(cat_input, timbre_P)
        cat_final = torch.cat([out_f0_mlp, out_l_mlp, out_mix_gru], dim=-1)

        out_final_mlp = self.final_mlp(cat_final)
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_mlp)

        # noise filter part
        noise_output = self.noise_head(out_final_mlp)

        return harmonic_output, noise_output, f0