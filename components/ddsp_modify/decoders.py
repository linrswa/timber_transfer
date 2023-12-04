import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCUB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv_1x1_input = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv_1x1_condition = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        
        # Using PyTorch's MultiheadAttention
        self.attention_block = nn.MultiheadAttention(embed_dim=in_ch, num_heads=8, batch_first=True)
        
        self.fc_after_attention = nn.Linear(in_ch, out_ch)  # to transform the output to desired dimension
        self.conv_1x1_output = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, condition):
        x_input = self.conv_1x1_input(x.transpose(1, 2))
        x_condition = self.conv_1x1_condition(condition.transpose(1, 2))
        
        # Applying attention
        attn_output, _ = self.attention_block(x, condition, x)
        x_attention = self.fc_after_attention(attn_output)
        
        mix = torch.cat([x_input, x_condition], dim=1)
        mix_tanh = torch.tanh(mix)
        mix_sigmoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_sigmoid
        mix_output = self.conv_1x1_output(mix_output)
        
        output = nn.LeakyReLU(0.2)(x_attention + mix_output.transpose(1, 2))
        return output 


class Decoder(nn.Module):
    def __init__(
        self,
        in_extract_size=32,
        mlp_layer=3,
        middle_embedding_size=256,
        n_harms = 101,
        noise_filter_bank = 65
        ):
        super().__init__()
        self.mlp_f0 = self.mlp(1, in_extract_size, mlp_layer)
        self.mlp_loudness = self.mlp(1, in_extract_size, mlp_layer)
        in_size = in_extract_size * 2

        self.tcrb_1 = TCUB(in_ch=in_size, out_ch=in_size * 2)
        self.tcrb_2 = TCUB(in_ch=in_size * 2, out_ch=in_size * 4)
        self.tcrb_3 = TCUB(in_ch=in_size * 4, out_ch=in_size * 8)
    
        self.mlp_final = self.mlp(in_size * 8 + in_size, middle_embedding_size, mlp_layer) 
        self.dense_harm = nn.Linear(middle_embedding_size, n_harms + 1)
        self.dense_noise = nn.Linear(middle_embedding_size, noise_filter_bank)
        
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