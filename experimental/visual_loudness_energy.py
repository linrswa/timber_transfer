# %%
import torch
import torchaudio.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append("..")
from data.dataset import NSynthDataset
from components.timbre_transformer.utils import extract_loudness, get_A_weight, extract_loudness_old
from tools.utils import cal_loudness_norm
from components.timbre_transformer.encoder import EngryEncoder

USE_MEAN_STD = True
FREQUENCY_WITH_CONFIDENCE = True
USE_SMOOTH = True
output_dir = "../output"
pt_file_dir = "../pt_file"

train_dataset = NSynthDataset(data_mode="valid", sr=16000, frequency_with_confidence=FREQUENCY_WITH_CONFIDENCE)

fn_with_path = random.choice(train_dataset.audio_list)
fn = fn_with_path.split("/")[-1][:-4]
_, s, l, f = train_dataset.getitem_by_filename(fn)
s = torch.Tensor(s).unsqueeze(dim=0)

energy_encoder = EngryEncoder()

A_weight = get_A_weight()
l_cal = extract_loudness(s, A_weight)
l_cal =  cal_loudness_norm(l_cal.view(-1))
n_fft = 1024
hop_length = 256
 
stft_transform = transforms.Spectrogram(
    n_fft=n_fft,
    hop_length=hop_length
)
stft_spec = stft_transform(s) + 0.1

mel_transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=128
)

mel_spec = mel_transform(s)
energy = energy_encoder(s).view(-1)
s = s.view(-1)
# Convert to log scale
log_mel_spec = torch.log(mel_spec + 1e-9)
log_stft_spec = torch.log(stft_spec + 1e-9) * 1000

# Normalize
log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
log_stft_spec = (log_stft_spec - log_stft_spec.min()) / (log_stft_spec.max() - log_stft_spec.min())

def plot_result():
    p = plt.plot
    plt.figure(figsize=(12, 8))
    plt.suptitle(fn)
    
    plt.subplot(511)
    p(s)
    plt.title("signal")
    
    plt.subplot(512)
    p(l)
    plt.xlim(0, 250)
    plt.title("loudness")
    
    plt.subplot(513)
    p(energy)
    plt.xlim(0, 250)
    plt.title("energy")
    
    plt.subplot(514)
    plt.imshow(log_mel_spec[0].detach().numpy(), aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.title("log mel spectrogram")

    plt.subplot(515)
    plt.imshow(log_stft_spec[0].detach().numpy(), aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    plt.title("log stft spectrogram")

    plt.tight_layout()
    plt.show()

plot_result()