#%%
import torch
import torch.nn as nn

class TCUB_before(nn.Module):
    def __init__(self, temporal, in_ch, out_ch):
        super().__init__()
        
        self.conv_1x1_input = nn.Conv1d(in_channels=in_ch, out_channels=temporal, kernel_size=1, stride=1, padding=0)
        self.conv_1x1_condition = nn.Conv1d(in_channels=in_ch, out_channels=temporal, kernel_size=1, stride=1, padding=0)
        
        # Using PyTorch's MultiheadAttention
        self.attention_block = nn.MultiheadAttention(embed_dim=in_ch, num_heads=8)
        
        self.fc_after_attention = nn.Linear(in_ch, out_ch)  # to transform the output to desired dimension
        self.conv_1x1_output = nn.Conv1d(in_channels=2*temporal, out_channels=out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, condition):
        x_input = self.conv_1x1_input(x.transpose(1, 2))
        x_condition = self.conv_1x1_condition(condition.transpose(1, 2))
        
        # Applying attention
        attn_output, _ = self.attention_block(x.transpose(0, 1), condition.transpose(0, 1), x.transpose(0, 1))
        x_attention = self.fc_after_attention(attn_output).transpose(0, 1)
        
        mix = torch.cat([x_input, x_condition], dim=1)
        mix_tanh = torch.tanh(mix)
        mix_sigmoid = torch.sigmoid(mix)
        mix_output = mix_tanh * mix_sigmoid
        mix_output = self.conv_1x1_output(mix_output)
        
        output = nn.LeakyReLU(0.2)(x_attention + mix_output.transpose(1, 2))
        return output 

class TCUB(nn.Module):
    def __init__(self, temporal, in_ch, out_ch):
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

# Test the model
model_attention = TCUB(temporal=250, in_ch=64, out_ch=128)
x = torch.rand(5, 250, 64)  # batch, temporal, feature_dim
condition = torch.rand(5, 250, 64)
output_attention = model_attention(x, condition)
output_attention.shape