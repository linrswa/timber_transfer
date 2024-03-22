#%%
import torch
import math
import torch.nn as nn
from .utils_blocks import DFBlock, TCUB, AttSubBlock, GateFusionBlock

from .utils import safe_divide

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
    def __init__(self, in_size=1, mlp_hidden_size=512, out_size=256, num_heads=8):
        super().__init__()
        self.input_mlp = MLP(in_size, mlp_hidden_size)
        self.gru = nn.GRU(mlp_hidden_size, out_size, num_layers=3, batch_first=True)
        self.cross_att_block = AttSubBlock(out_size, num_heads)
    
    def forward(self, x, condition):
        x = self.input_mlp(x)
        x, _ = self.gru(x)
        x = self.cross_att_block(x, condition)
        return x

class AmpStack(nn.Module):
    def __init__(self, emb_dim=8):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(2, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_dim, 1),
            )
    
    def forward(self, amp, loudness):
        amp_mix = torch.cat([amp, loudness], dim=-1)
        amp_att = self.stack(amp_mix)
        return modified_sigmoid(amp + amp_att)


class HarmonicHead(nn.Module):
    def __init__(self, in_size, timbre_emb_size, n_harms):
        super().__init__()
        self.dense_harm = nn.Linear(in_size, n_harms+1)
        self.dfblock1 = DFBlock(n_harms, timbre_emb_size, affine_dim=n_harms, out_layer_mlp=True)
        self.dfblock2 = DFBlock(n_harms, timbre_emb_size, affine_dim=n_harms, out_layer_mlp=True)
        self.relu = nn.LeakyReLU(0.2)

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
        global_amp = modified_sigmoid(global_amp)

        n_harm_dis = modified_sigmoid(n_harm_dis)
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
        in_extract_size=256,
        timbre_emb_size=256,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.f0_input_block = InputAttBlock(in_size=1, out_size=in_extract_size)
        self.loudness_input_block = InputAttBlock(in_size=1, out_size=in_extract_size)
        self.mix_gru = nn.GRU(in_extract_size * 3, final_embedding_size, num_layers=3, batch_first=True)
        self.self_att= AttSubBlock(final_embedding_size)
        self.final_mlp = MLP(final_embedding_size * 2, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, timbre_emb_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        out_f0_input = self.f0_input_block(f0, timbre_emb)
        out_loudness_input = self.loudness_input_block(loudness, timbre_emb)
        mix = torch.cat([out_f0_input, out_loudness_input, timbre_emb.expand_as(out_f0_input)], dim=-1)
        out_mix_gru = self.mix_gru(mix)[0]
        out_self_att = self.self_att(out_mix_gru, out_mix_gru)
        
        out_cat_f0_loudness = torch.cat([out_self_att, out_f0_input, out_loudness_input], dim=-1)
        out_final_mlp = self.final_mlp(out_cat_f0_loudness)
        
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_mlp, timbre_emb)

        # noise filter part
        noise_output = self.noise_head(out_final_mlp)

        return harmonic_output, noise_output

