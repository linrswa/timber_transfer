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
        in_extract_size=128,
        timbre_emb_size=128,
        final_embedding_size=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.input_f0 = InputAttBlock(in_size=1, out_size=in_extract_size)
        self.input_loudness = InputAttBlock(in_size=1, out_size=in_extract_size)
        gru_in_size = in_extract_size * 2
        gru_out_size = in_extract_size
        self.mix_gru = nn.GRU(gru_in_size, gru_out_size, num_layers=1, batch_first=True)
        self.self_att_1 = AttSubBlock(gru_out_size)

        in_size = in_extract_size
        self.condition_proj_1 = nn.Linear(timbre_emb_size, in_size * 2)
        self.condition_proj_2 = nn.Linear(timbre_emb_size, in_size * 4)
        self.tcub_1 = TCUB(in_size)
        self.tcub_2 = TCUB(in_size * 2)
        self.self_att_2 = AttSubBlock(in_size * 2)
        self.cross_att_1 = AttSubBlock(in_size * 2)
        self.cross_att_2 = AttSubBlock(in_size * 4)

        final_size = in_size * 4 + in_size * 2 
        self.final_self_att = AttSubBlock(final_size) 
        self.final_self_att_proj = nn.Linear(final_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, timbre_emb_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        
        out_input_f0 = self.input_f0(f0)
        out_input_loudness = self.input_loudness(loudness)
        out_cat_mlp = torch.cat([out_input_f0, out_input_loudness], dim=-1)
        out_mix_gru = self.mix_gru(out_cat_mlp)[0]
        out_cat_mlp = self.self_att_1(out_mix_gru, out_mix_gru)

        timbre_emb_1 = self.condition_proj_1(timbre_emb)
        timbre_emb_2 = self.condition_proj_2(timbre_emb)
        out_timbre_fusion_1 = self.tcub_1(out_cat_mlp, timbre_emb)
        out_timbre_fusion_1 = self.self_att_2(out_timbre_fusion_1, out_timbre_fusion_1)
        out_timbre_fusion_1 = self.cross_att_1(out_timbre_fusion_1, timbre_emb_1)
        out_timbre_fusion_2 = self.tcub_2(out_timbre_fusion_1, timbre_emb_1)
        out_timbre_fusion_2 = self.cross_att_2(out_timbre_fusion_2, timbre_emb_2)

        out_cat_f0_loudness = torch.cat([out_timbre_fusion_2, out_input_f0, out_input_loudness], dim=-1)
        out_final_self_att = self.final_self_att(out_cat_f0_loudness, out_cat_f0_loudness)
        out_final_self_att = self.final_self_att_proj(out_final_self_att)
        
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_self_att, timbre_emb)

        # noise filter part
        noise_output = self.noise_head(out_final_self_att)

        return harmonic_output, noise_output

