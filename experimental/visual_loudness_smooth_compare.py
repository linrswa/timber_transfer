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
pt_file = f"{pt_file_dir}/base_8_generator_best_3.pt"
ae.load_state_dict(torch.load(f"{pt_file}", map_location="cpu"))

synthsizer = HarmonicOscillator().to("cpu")
synthsizer_smooth = HarmonicOscillator(is_smooth=True).to("cpu")
noise_filter = NoiseFilter().to("cpu")

harmonic_head_output, f0, noise_head_output = ae(s, l_mod, f0)
add = synthsizer(harmonic_head_output, f0)
add_smooth = synthsizer_smooth(harmonic_head_output, f0)
sub = noise_filter(noise_head_output)
rec = add + sub 
rec_smooth = add_smooth + sub
global_amp = harmonic_head_output[1]

A_weight = get_A_weight()
l = extract_loudness(s, A_weight)
rec_l = extract_loudness(rec.squeeze(dim=-1), A_weight)
rec_l_smooth = extract_loudness(rec_smooth.squeeze(dim=-1), A_weight)

s = s.view(-1).numpy()
rec = rec.view(-1).detach().numpy()
rec_smooth = rec_smooth.view(-1).detach().numpy()

l = l.view(-1).detach().numpy()
rec_l = rec_l.view(-1).detach().numpy()
rec_l_smooth = rec_l_smooth.view(-1).detach().numpy()
l, rec_l, rec_l_smooth = cal_loudness_norm(l), cal_loudness_norm(rec_l), cal_loudness_norm(rec_l_smooth)

global_amp = global_amp.view(-1).detach().numpy()

def plot_result():
    p = plt.plot
    plt.suptitle(fn[0])
    plt.subplot(331)
    p(s)
    plt.title("ori")
    plt.subplot(332)
    p(rec)
    plt.title("rec")
    plt.subplot(333)
    p(rec_smooth)
    plt.title("rec_smooth")
    plt.subplot(334)
    p(l)
    plt.title("l")
    plt.subplot(335)
    p(rec_l)
    plt.title("rec_l")
    plt.subplot(336)
    diff_l = abs(l - rec_l)
    p(diff_l, color="red")
    plt.title(f"diff_loudness {diff_l.mean(): .3f}")
    plt.subplot(337)
    p(l)
    plt.title("l")
    plt.subplot(338)
    p(rec_l_smooth)
    plt.title("rec_l_smooth")
    plt.subplot(339)
    diff_fix_l_smooth = abs(l[:-1] - rec_l_smooth[:-1])
    p(diff_fix_l_smooth, color="red")
    plt.title(f"diff_fix_loudness_smooth{diff_fix_l_smooth.mean(): .3f}")
    plt.tight_layout()

plot_result()