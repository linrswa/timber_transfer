# %%
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from glob import glob
import os 

import sys
sys.path.append("..")
from data.dataset import NSynthDataset
from components.timbre_transformer.TimberTransformer import TimbreTransformer 
from components.timbre_transformer.TimbreFusionAE import TimbreFusionAE
from components.timbre_transformer.component import HarmonicOscillator, NoiseFilter
from components.timbre_transformer.utils import extract_loudness, get_A_weight, get_extract_pitch_needs, extract_pitch
from tools.utils import cal_loudness_norm, seperate_f0_confidence, mask_f0_with_confidence
from tools.utils import get_loudness_mask

USE_MEAN_STD = True
FREQUENCY_WITH_CONFIDENCE = True
USE_SMOOTH = True
output_dir = "../output"
pt_file_dir = "../pt_file"

train_dataset = NSynthDataset(data_mode="valid", sr=16000, frequency_with_confidence=FREQUENCY_WITH_CONFIDENCE)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f0_with_confidence = next(iter(train_loader)) 

if FREQUENCY_WITH_CONFIDENCE:
    f0, _ = seperate_f0_confidence(f0_with_confidence)

if USE_MEAN_STD:
    l_mod = cal_loudness_norm(l)

ae = TimbreFusionAE()
pt_file = f"{pt_file_dir}/train9_generator_best_1.pt"
ae.load_state_dict(torch.load(f"{pt_file}"))

synthsizer = HarmonicOscillator()
noise_filter = NoiseFilter()

harmonic_head_output, f0, noise_head_output = ae(s, l_mod, f0)
add = synthsizer(harmonic_head_output, f0)
sub = noise_filter(noise_head_output)
rec = add + sub 
global_amp = harmonic_head_output[1]

A_weight = get_A_weight()
rec_l = extract_loudness(rec.squeeze(dim=-1), A_weight)
device, cr_model, m_sec = get_extract_pitch_needs(device="cpu")
rec_f0_confidence = extract_pitch(rec.squeeze(dim=-1), device=device, cr=cr_model, m_sec=m_sec, with_confidence=True)
rec_f0 = mask_f0_with_confidence(rec_f0_confidence, threshold=0.85)
f0 = mask_f0_with_confidence(f0_with_confidence, threshold=0.85)

s = s.view(-1).numpy()
l = l.view(-1)
f0 = f0.view(-1)

rec = rec.view(-1).detach().numpy()
rec_l = rec_l.view(-1).detach().numpy()[:-1]
rec_f0 = rec_f0.view(-1).numpy()

add_l = extract_loudness(add.squeeze(dim=-1), A_weight)
sub_l = extract_loudness(sub.squeeze(dim=-1), A_weight)
add = add.view(-1).detach().numpy()
sub = sub.view(-1).detach().numpy()
add_l = add_l.view(-1).detach().numpy()[:-1]
sub_l = sub_l.view(-1).detach().numpy()[:-1]
add_l, sub_l = cal_loudness_norm(add_l), cal_loudness_norm(sub_l)
l, rec_l = cal_loudness_norm(l), cal_loudness_norm(rec_l)

global_amp = global_amp.view(-1).detach().numpy()

loudness_mask = get_loudness_mask(s).reshape(-1)

def plot_result(s, rec, fn, rec_l, l):
    p = plt.plot
    wf.write(f"{output_dir}/tmp/ori.wav", 16000, s)
    wf.write(f"{output_dir}/tmp/rec.wav", 16000, rec)
    wf.write(f"{output_dir}/tmp/add.wav", 16000, add)
    wf.write(f"{output_dir}/tmp/sub.wav", 16000, sub)
    plt.suptitle(fn[0])
    plt.subplot(331)
    p(s)
    plt.title("ori")
    plt.subplot(332)
    p(rec)
    plt.title("rec")
    plt.subplot(334)
    p(l)
    plt.title("ori_loudness")
    plt.subplot(333)
    diff_l = abs(l - rec_l)
    p(diff_l, color="red")
    plt.title(f"diff_loudness {diff_l.mean(): .3f}")
    plt.subplot(335)
    p(rec_l)
    plt.title("rec_loudness")
    plt.subplot(336)
    diff_l = abs(l - rec_l) * loudness_mask
    p(diff_l, color="red")
    plt.title(f"diff_fix_loudness {diff_l.mean(): .3f}")
    plt.subplot(337)
    p(add_l)
    plt.title("add_loudness")
    plt.subplot(338)
    p(sub_l)
    plt.title("sub_loudness")
    plt.subplot(339)
    p(global_amp)
    plt.title("global_amp")
    plt.tight_layout()

plot_result(s, rec, fn, rec_l, l)