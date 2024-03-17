# %%
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

import sys
sys.path.append("..")
from data.dataset import NSynthDataset
from components.timbre_transformer.TimbreFusionAE import TimbreFusionAE
from components.timbre_transformer.component import HarmonicOscillator, NoiseFilter
from components.timbre_transformer.utils import extract_loudness, get_A_weight, extract_loudness_old
from tools.utils import cal_loudness_norm, get_loudness_mask
from tools.utils import seperate_f0_confidence

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

ae = TimbreFusionAE().to("cpu")
pt_file = f"{pt_file_dir}/train10_generator_best_40.pt"
ae.load_state_dict(torch.load(f"{pt_file}", map_location="cpu"))

synthsizer = HarmonicOscillator().to("cpu")
noise_filter = NoiseFilter().to("cpu")

harmonic_head_output, f0, noise_head_output = ae(s, l_mod, f0)
add = synthsizer(harmonic_head_output, f0)
sub = noise_filter(noise_head_output)
rec = add + sub 
global_amp = harmonic_head_output[1]

A_weight = get_A_weight()
l_old = extract_loudness_old(s, A_weight)
rec_l_old = extract_loudness_old(rec.squeeze(dim=-1), A_weight)
l = extract_loudness(s, A_weight)
rec_l = extract_loudness(rec.squeeze(dim=-1), A_weight)

s = s.view(-1).numpy()

rec = rec.view(-1).detach().numpy()
rec_l_old = rec_l_old.view(-1).detach().numpy()

l_old = l_old.view(-1).detach().numpy()
l = l.view(-1).detach().numpy()
rec_l = rec_l.view(-1).detach().numpy()
l_old, rec_l_old = cal_loudness_norm(l_old), cal_loudness_norm(rec_l_old)
l, rec_l = cal_loudness_norm(l), cal_loudness_norm(rec_l)

global_amp = global_amp.view(-1).detach().numpy()

loudness_mask = get_loudness_mask(s).reshape(-1)

def plot_result():
    p = plt.plot
    plt.suptitle(fn[0])
    plt.subplot(331)
    p(s)
    plt.title("ori")
    plt.subplot(332)
    p(rec)
    plt.title("rec")
    plt.subplot(334)
    p(l_old)
    plt.title("l_old")
    plt.subplot(333)
    diff_l_old = abs(l_old - rec_l_old)
    p(diff_l_old, color="red")
    plt.title(f"diff_loudness_old {diff_l_old.mean(): .3f}")
    plt.subplot(335)
    p(rec_l_old)
    plt.title("rec_l_old")
    plt.subplot(336)
    diff_fix_l_old = abs(l_old[:-1] - rec_l_old[:-1]) * loudness_mask
    p(diff_fix_l_old, color="red")
    plt.title(f"diff_fix_loudness_old {diff_fix_l_old.mean(): .3f}")
    plt.subplot(337)
    p(l)
    plt.title("l")
    plt.subplot(338)
    p(rec_l)
    plt.title("rec_l")
    plt.subplot(339)
    diff_l = abs(l - rec_l)
    p(diff_l, color="red")
    plt.title(f"diff_loudness {diff_l.mean(): .3f}")
    plt.tight_layout()

plot_result()