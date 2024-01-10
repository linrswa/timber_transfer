import math
import torch
import torch.nn as nn
from .utils_blocks import UpFusionBlock, DFBlock

# force the amplitudes, harmonic distributions, and filtered noise magnitudes 
# to be non-negative by applying a sigmoid nonlinearity to network outputs.
def modified_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

class HarmonicHead(nn.Module):
    def __init__(self, in_size, timbre_emb_size, n_harms):
        super().__init__()
        head_dim_size = n_harms + 1
        self.dense_harm = nn.Linear(in_size, head_dim_size)
        self.dfblock1 = DFBlock(head_dim_size, timbre_emb_size, affine_dim=head_dim_size, out_layer_mlp=True)
        self.dfblock2 = DFBlock(head_dim_size, timbre_emb_size, affine_dim=head_dim_size, out_layer_mlp=True)

    def forward(self, out_mlp_final, timbre_emb):
        out_dense_harm = self.dense_harm(out_mlp_final)
        df_out = self.dfblock1(out_dense_harm, timbre_emb)
        df_out = self.dfblock2(df_out, timbre_emb)
        out_dense_harm = out_dense_harm + df_out

        # out_dense_harmonic output -> 1(global_amplitude) + n_harmonics 
        global_amp = modified_sigmoid(out_dense_harm[..., :1])
        n_harm_amps = out_dense_harm[..., 1:]
        # n_harm_amps /= n_harm_amps.sum(-1, keepdim=True) # not every element >= 0
        n_harm_amps_norm = nn.functional.softmax(n_harm_amps, dim=-1)
        harm_amp_distribution = global_amp * n_harm_amps_norm

        return harm_amp_distribution, global_amp


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
        self.dense_noise = nn.Linear(final_embedding_size, noise_filter_bank)
        
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
        harm_amp_distribution = self.harmonic_head(out_mlp_final, timbre_emb)

        # noise filter part
        out_dense_noise = self.dense_noise(out_mlp_final)
        noise_filter_bank = modified_sigmoid(out_dense_noise)

        return harm_amp_distribution, noise_filter_bank


    @staticmethod
    def mlp(in_size, hidden_size, n_layers):
        channels = [in_size] + (n_layers) * [hidden_size]
        net = []
        for i in range(n_layers):
            net.append(nn.Linear(channels[i], channels[i + 1]))
            net.append(nn.LayerNorm(channels[i+1]))
            net.append(nn.LeakyReLU())
        return nn.Sequential(*net)

