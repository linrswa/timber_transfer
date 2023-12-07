import torch
from torch import nn 
from collections import OrderedDict

class Affine(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.embedding_dim = emb_dim
        self.fc_alpha = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(emb_dim, emb_dim)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(emb_dim, 1)),
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(emb_dim, emb_dim)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(emb_dim, 1)),
        ]))

    def _initialize(self):
        nn.init.zeros_(self.fc_alpha.linear2.weight.data)
        nn.init.zeros_(self.fc_alpha.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, condition_emb):
        weight = self.fc_alpha(condition_emb)
        bias = self.fc_beta(condition_emb)

        if x.dim() != 1:
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)

        return x * weight + bias


class DFBlock(nn.Module):
    def __init__(self, in_ch, emb_dim):
        super().__init__()
        self.affine1 = Affine(emb_dim)
        self.affine2 = Affine(emb_dim)
        self.conv = nn.Conv1d(in_ch, in_ch, kernel_size=1)

    def forward(self, x, condition_emb):
        x = self.affine1(x, condition_emb)
        x = nn.LeakyReLU(0.2)(x)
        x = self.affine2(x, condition_emb)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        hidden_ch = in_ch * 4
        out_ch = in_ch * 2
        self.conv_to_4_times = nn.Conv1d(in_ch, hidden_ch, kernel_size=1)
        self.ln = nn.LayerNorm((hidden_ch, 250))
        self.conv_out = nn.Conv1d(out_ch, out_ch, kernel_size=1)
    
    def GLU(self, x):
        # x: [B, emb, T]
        # x_sigmoid: [B, emb/2, T], x_tanh: [B, emb/2, T]
        emb_dim = x.size(1)
        x_sigmoid, x_tanh = x.split(emb_dim // 2, dim=1)
        out = torch.sigmoid(x_sigmoid) * torch.tanh(x_tanh)
        return out

    def forward(self, x):
        # input: [B, emb, T]
        x = self.conv_to_4_times(x)
        x = self.ln(x)
        x = self.GLU(x)
        x = self.conv_out(x)
        x = nn.LeakyReLU(0,2)(x)
        return x
    

class UpFusionBlock(nn.Module):
    def __init__(self, in_ch, emb_dim):
        super().__init__()
        out_ch = in_ch * 2
        self.upblock = UpBlock(in_ch)
        self.dfblock1 = DFBlock(out_ch, emb_dim)
        self.dfblock2 = DFBlock(out_ch, emb_dim)
        
    def forward(self, x, condition_emb):
        x_up = self.upblock(x)
        condition_out = self.dfblock1(x_up, condition_emb)
        condition_out = self.dfblock2(condition_out, condition_emb)
        out = x_up + condition_out
        return out

if __name__ == "__main__":
    x = torch.randn(4, 256, 250)
    emb = torch.randn(4, 128)
    upblock = UpFusionBlock(256, 128)
    out = upblock(x, emb)
    print(out.shape)