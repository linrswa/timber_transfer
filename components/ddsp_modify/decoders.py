import math
import torch
import torch.nn as nn
from .utils_blocks import TCUB, UpFusionBlock

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
        in_size = in_extract_size * 2

        self.timbre_gru_1 = nn.GRU(timbre_emb_size, timbre_emb_size, batch_first=True)
        self.timbre_gru_2 = nn.GRU(timbre_emb_size, timbre_emb_size, batch_first=True)
        self.timbre_gru_3 = nn.GRU(timbre_emb_size, timbre_emb_size, batch_first=True)

        self.upfusionblock_1 = UpFusionBlock(in_ch=in_size, emb_dim=timbre_emb_size) # in_size = 128
        self.upfusionblock_2 = UpFusionBlock(in_ch=in_size*2, emb_dim=timbre_emb_size) # in_size = 256
        self.upfusionblock_3 = UpFusionBlock(in_ch=in_size*4, emb_dim=timbre_emb_size) # in_size = 512
    
        self.mlp_final = self.mlp(in_size * 8 + in_size, final_embedding_size, mlp_layer) 
        self.dense_harm = nn.Linear(final_embedding_size, n_harms + 1)
        self.dense_noise = nn.Linear(final_embedding_size, noise_filter_bank)
        
    def forward(self, f0, loudness, timbre_emb):
        out_mlp_f0 = self.mlp_f0(f0)
        out_mlp_loudness = self.mlp_loudness(loudness)
        out_cat_mlp = torch.cat([out_mlp_f0, out_mlp_loudness], dim=-1)

        out_before_up = out_cat_mlp.transpose(1, 2).contiguous()
        timbre_emb_1, hidden_state_1 = self.timbre_gru_1(timbre_emb)
        out_upfb_1 = self.upfusionblock_1(out_before_up, timbre_emb_1)
        timbre_emb_2, hidden_statte_2 = self.timbre_gru_2(timbre_emb_1, hidden_state_1)
        out_upfb_2 = self.upfusionblock_2(out_upfb_1, timbre_emb_2)
        timbre_emb_3, _ = self.timbre_gru_3(timbre_emb_2, hidden_statte_2)
        out_upfb_3 = self.upfusionblock_3(out_upfb_2, timbre_emb_3)
        out_after_up = out_upfb_3.transpose(1, 2).contiguous()

        out_cat_f0_loudness = torch.cat([out_after_up, out_mlp_f0, out_mlp_loudness], dim=-1)
        out_mlp_final = self.mlp_final(out_cat_f0_loudness)
        
        # harmonic part
        out_dense_harm = self.dense_harm(out_mlp_final)
        # out_dense_harmonic output -> 1(global_amplitude) + n_harmonics 
        global_amp = self.modified_sigmoid(out_dense_harm[..., :1])
        n_harm_amps = out_dense_harm[..., 1:]
        # n_harm_amps /= n_harm_amps.sum(-1, keepdim=True) # not every element >= 0
        n_harm_amps_norm = nn.functional.softmax(n_harm_amps, dim=-1)
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

