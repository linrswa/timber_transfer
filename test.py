#%%
import torch
import torch.nn as nn
import torchcrepe
import torchaudio

class F0Encoder(nn.Module):
    """基頻編碼器，使用預訓練的 CREPE 模型來提取基頻"""
    def __init__(self, sample_rate=16000, hop_length=256):
        super(F0Encoder, self).__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def forward(self, signal):
        if signal.dim() == 2:  # Already (batch_size, time_steps)
            signal = signal
        elif signal.dim() == 3:  # (batch_size, 1, time_steps)
            signal = signal.squeeze(1)  # Convert to (batch_size, time_steps)
        
        # 使用 CREPE 預測基頻，返回 (batch_size, time_steps) 形狀
        f0 = torchcrepe.predict(signal, self.sample_rate, self.hop_length, model="full", device=signal.device)
        
        # 確保輸出的 time_steps 是 250
        f0 = f0[:, :250]
        return f0.unsqueeze(-1)  # (batch_size, 250, 1) 擴張一個維度


class LoudnessEncoder(nn.Module):
    """響度編碼器，使用 A-weighting 的功率譜提取響度特徵"""
    
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256):
        super(LoudnessEncoder, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.a_weighting = self._create_a_weighting_filter()
    
    def _create_a_weighting_filter(self):
        def a_weighting(spectrum):
            freq_bins = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
            a_weighting_curve = (
                (freq_bins ** 2) * (10 ** (0.1 * (freq_bins - 20) / 20))
            ) / (freq_bins ** 2 + 20.6 ** 2)
            a_weighting_curve /= a_weighting_curve.max()  # 標準化
            return spectrum * a_weighting_curve.unsqueeze(0)  # 應用 A-weighting

        return a_weighting

    def forward(self, signal):
        # 計算功率譜
        power_spectrum = torch.abs(torch.fft.fft(signal, n=self.n_fft))**2
        power_spectrum = power_spectrum[:, :self.n_fft // 2 + 1]  # 取前半部分
        
        # 應用 A-weighting
        weighted_spectrum = self.a_weighting(power_spectrum)

        # 對數縮放
        loudness = 10 * torch.log10(weighted_spectrum + 1e-10)

        # 計算響度的均值和標準差
        mean = loudness.mean()
        std = loudness.std()

        # 中心化處理
        loudness = (loudness - mean) / std

        # 確保響度的時間步長與其他編碼器一致
        # 假設原始的時間步長是 len(signal) // hop_length
        time_steps = signal.size(1) // self.hop_length
        loudness = loudness.view(1, -1)  # (batch_size, freq_bins)
        loudness = loudness[:time_steps]  # 裁剪到時間步長

        # 確保輸出形狀為 (batch_size, 250, 1)
        if loudness.size(1) < 250:
            # 使用填充
            loudness = torch.cat([loudness, torch.zeros(1, 250 - loudness.size(1))], dim=1)
        else:
            loudness = loudness[:, :250]

        return loudness.unsqueeze(-1)  # 返回形狀 (batch_size, 250, 1)

class ZEncoder(nn.Module):
    """潛在向量編碼器，使用 MFCC 和 GRU 來提取潛在向量"""
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mfcc=30, n_mels=128, gru_units=512, z_units=16):
        super(ZEncoder, self).__init__()
        
        self.mfcc_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length,
                'f_min': 20.0,
                'f_max': 8000.0,
            }
        )
        
        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.gru = nn.GRU(input_size=n_mfcc, hidden_size=gru_units, batch_first=True)
        self.linear = nn.Linear(gru_units, z_units)

    def forward(self, signal):
        signal = signal.squeeze(dim=1)
        mfcc = self.mfcc_extractor(signal)
        mfcc = self.norm(mfcc)
        mfcc = mfcc.permute(0, 2, 1).contiguous()  # (batch_size, time_steps, n_mfcc)
        gru_output, _ = self.gru(mfcc)
        z = self.linear(gru_output)
        z = z[:, :250]
        return z

class CombinedEncoder(nn.Module):
    """將基頻、響度和潛在向量編碼器結合"""
    def __init__(self, sample_rate=16000, hop_length=256, gru_units=512, z_units=16):
        super(CombinedEncoder, self).__init__()
        self.f0_encoder = F0Encoder(sample_rate, hop_length)
        self.l_encoder = LoudnessEncoder(sample_rate, hop_length)
        self.z_encoder = ZEncoder(gru_units=gru_units, z_units=z_units)

    def forward(self, signal):
        f0 = self.f0_encoder(signal)
        l = self.l_encoder(signal)
        z = self.z_encoder(signal)
        return f0, z, l

# 測試模型
# encoder = CombinedEncoder()
# encoder_input = torch.randn(1, 64000)  # (batch_size=1, time_steps=64000)
# f0, l, z = encoder(encoder_input)

# print(f"f0: {f0.shape}, z: {z.shape}, l: {l.shape}")

def f0_method_compare():
    from components.timbre_transformer.utils import get_extract_pitch_needs, extract_pitch

    encoder_input, _ = torchaudio.load("sound.wav")

    # 測試 extract_pitch 和 F0Encoder 是否表現一致
    device, cr, m_sec = get_extract_pitch_needs(device="cpu")
    pitch = extract_pitch(encoder_input, device, cr, m_sec)

    # 使用 F0Encoder 提取基頻
    f0_encoder = F0Encoder()
    f0_output = f0_encoder(encoder_input)

    # 比較兩者的輸出
    print(f"Extracted pitch shape: {pitch.unsqueeze(dim=-1)[:,:-1,:].shape}, F0Encoder output shape: {f0_output.shape}")
    print(f"Difference: {torch.abs(pitch.unsqueeze(dim=-1)[:,:-1,:] - f0_output).sum()}")
    import matplotlib.pyplot as plt

    # Plot the extracted pitch
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(pitch.squeeze().cpu().numpy(), label='Extracted Pitch')
    plt.title('Extracted Pitch')
    plt.xlabel('Time Steps')
    plt.ylabel('Frequency (Hz)')
    plt.legend()

    # Plot the F0Encoder output
    plt.subplot(2, 1, 2)
    plt.plot(f0_output.squeeze().cpu().numpy(), label='F0Encoder Output', color='orange')
    plt.title('F0Encoder Output')
    plt.xlabel('Time Steps')
    plt.ylabel('Frequency (Hz)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def loudness_method_compare():
    from components.timbre_transformer.utils import get_A_weight, extract_loudness
    encoder_input, _ = torchaudio.load("sound.wav")

    # 測試 extract_loudness 和 LoudnessEncoder 是否表現一致
    a_weighting = get_A_weight()
    loudness = extract_loudness(encoder_input, a_weighting)

    # 使用 LoudnessEncoder 提取響度
    l_encoder = LoudnessEncoder()
    l_output = l_encoder(encoder_input)

    # 比較兩者的輸出
    print(f"Extracted loudness shape: {loudness.unsqueeze(dim=-1)[:,:-1,:].shape}, LoudnessEncoder output shape: {l_output.shape}")
    print(f"Difference: {torch.abs(loudness.unsqueeze(dim=-1)[:,:-1,:] - l_output).sum()}")
    import matplotlib.pyplot as plt

    # Plot the extracted loudness
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(loudness.squeeze().cpu().numpy(), label='Extracted Loudness')
    plt.title('Extracted Loudness')
    plt.xlabel('Time Steps')
    plt.ylabel('Loudness (dB)')
    plt.legend()

    # Plot the LoudnessEncoder output
    plt.subplot(2, 1, 2)
    plt.plot(l_output.squeeze().cpu().numpy(), label='LoudnessEncoder Output', color='orange')
    plt.title('LoudnessEncoder Output')
    plt.xlabel('Time Steps')
    plt.ylabel('Loudness (dB)')
    plt.legend()

    plt.tight_layout()
    plt.show()

loudness_method_compare()