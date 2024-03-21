#%%
import torch
import math
import torch.nn as nn
from .utils_blocks import DFBlock, TCUB, AttSubBlock, GateFusionBlock

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
    def __init__(self, in_size=1, hidden_size=32, out_size=64):
        super().__init__()
        if hidden_size % 8 == 0:
            num_heads = 8
        else:
            num_heads = 1
        self.input_linear = nn.Linear(in_size, hidden_size)
        self.first_att_block = AttSubBlock(hidden_size, num_heads)
        self.out_linear = nn.Linear(hidden_size, out_size)
        self.out_att_block = AttSubBlock(out_size, num_heads)
    
    def forward(self, x):
        x = self.input_linear(x)
        x = self.first_att_block(x, x)
        x = self.out_linear(x)
        x = self.out_att_block(x, x)
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

        # n_harm_amps /= n_harm_amps.sum(-1, keepdim=True) # not every element >= 0
        n_harm_dis_norm = nn.functional.softmax(n_harm_dis, dim=-1)

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
        mix_hidden_size=512,
        timbre_emb_size=128,
        final_embedding_size=512,
        gru_units=512,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.input_f0 = MLP(1, in_extract_size)
        self.input_loudness =  MLP(1, in_extract_size)
        self.input_mixer = nn.GRU(in_extract_size * 2, mix_hidden_size)
    
        stage_1_size = mix_hidden_size
        self.gfb_stage_1_1 = GateFusionBlock(stage_1_size)
        self.gfb_stage_1_2 = GateFusionBlock(stage_1_size)
        self.mlp_1 = MLP(stage_1_size, stage_1_size * 2) 
        self.stage_1_gru = nn.GRU(stage_1_size * 2, gru_units, batch_first=True)

        stage_2_size = stage_1_size
        self.gfb_stage_2_1 = GateFusionBlock(stage_2_size)
        self.gfb_stage_2_2 = GateFusionBlock(stage_2_size)
        self.mlp_2 = MLP(stage_2_size, stage_2_size) 
        self.stage_2_gru = nn.GRU(stage_1_size, gru_units, batch_first=True)

        final_size = final_embedding_size + in_extract_size * 2
        self.final_mlp = MLP(final_size, final_embedding_size)
        self.harmonic_head = HarmonicHead(final_embedding_size, timbre_emb_size, n_harms)
        self.noise_head = NoiseHead(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        
        out_input_f0 = self.input_f0(f0)
        out_input_loudness = self.input_loudness(loudness)
    
        out_cat_mlp = torch.cat([out_input_f0, out_input_loudness], dim=-1)
        out_mlp_mixer, _ = self.input_mixer(out_cat_mlp)

        out_stage_1 = self.gfb_stage_1_1(out_mlp_mixer, timbre_emb)
        out_stage_1 = self.gfb_stage_1_2(out_stage_1, timbre_emb)
        out_stage_1 = self.mlp_1(out_stage_1)
        out_stage_1, _ = self.stage_1_gru(out_stage_1)

        out_stage_2 = self.gfb_stage_2_1(out_stage_1, timbre_emb)
        out_stage_2 = self.gfb_stage_2_2(out_stage_1, timbre_emb)
        out_stage_2 = self.mlp_2(out_stage_1)
        out_stage_2, _ = self.stage_2_gru(out_stage_1)

        out_cat_f0_loudness = torch.cat([out_stage_2, out_input_f0, out_input_loudness], dim=-1)
        out_final_mlp = self.final_mlp(out_cat_f0_loudness)
        
        # harmonic part
        harmonic_output = self.harmonic_head(out_final_mlp, timbre_emb)

        # noise filter part
        noise_output = self.noise_head(out_final_mlp)

        return harmonic_output, noise_output

