import torch
import torch.nn as nn

class TCUB(nn.Module):
    def __init__(self, in_ch, num_heads=8):
        super().__init__()
        out_ch = in_ch * 2
        self.conv_1x1_input = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.conv_1x1_condition = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        
        # Using PyTorch's MultiheadAttention
        self.attention_block = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads, batch_first=True)
        
        self.fc_after_attention = nn.Linear(in_ch, out_ch)  # to transform the output to desired dimension
        self.conv_1x1_output = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, condition):
        x_input = self.conv_1x1_input(x.transpose(1, 2))
        x_condition = self.conv_1x1_condition(condition.transpose(1, 2))
        
        # Applying attention
        attn_output, _ = self.attention_block(x, condition, condition)
        x_attention = self.fc_after_attention(attn_output)
        
        mix = torch.cat([x_input, x_condition], dim=1)
        mix_tanh = torch.tanh(mix)
        mix_sigmoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_sigmoid
        mix_output = self.conv_1x1_output(mix_output)
        
        output = nn.LeakyReLU(0.2)(x_attention + mix_output.transpose(1, 2))
        return output 

class AttSubBlock(nn.Module):
    def __init__(self, in_ch, num_heads=8):
        super().__init__()
        self.self_att = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads, batch_first=True)
        self.self_att_norm = nn.LayerNorm(in_ch)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_ch * 2, in_ch),
            nn.LeakyReLU(0.2)
        )
        self.mlp_norm = nn.LayerNorm(in_ch)
    
    def forward(self, x, condition):
        att_out, _ = self.self_att(x, condition, condition)
        att_out = self.self_att_norm(x + att_out)
        fc_out = self.mlp(att_out)
        out = self.mlp_norm(att_out + fc_out)
        return out
    
class TimbreAttFusionBlock(nn.Module):
    def __init__(self, in_emb, timbre_emb, num_heads=8) -> None:
        super().__init__()
        out_emb = in_emb * 2
        self.input_fc = nn.Linear(in_emb, out_emb)
        self.input_self_att = AttSubBlock(out_emb, num_heads)
        self.timbre_fc = nn.Linear(timbre_emb, out_emb)
        self.timbre_self_att = AttSubBlock(out_emb, num_heads)
        self.timbre_fusion_att = AttSubBlock(out_emb, num_heads)
    
    def forward(self, x, timbre_emb):
        x = self.input_fc(x)
        x = self.input_self_att(x, x)
        timbre_emb = self.timbre_fc(timbre_emb)
        timbre_emb = self.timbre_self_att(timbre_emb, timbre_emb)
        timbre_fusion_emb = self.timbre_fusion_att(x, timbre_emb)
        return timbre_fusion_emb