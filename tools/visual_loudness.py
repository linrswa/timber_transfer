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
from components.timbre_transformer.component import EnhanceHarmonicOscillator 
from components.timbre_transformer.utils import extract_loudness, get_A_weight, get_extract_pitch_needs, extract_pitch
from tools.utils import cal_loudness_norm, seperate_f0_confidence, mask_f0_with_confidence

output_dir = "../output"
pt_file_dir = "../pt_file"

train_dataset = NSynthDataset(data_mode="valid", sr=16000, frequency_with_confidence=True)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f0_with_confidence = next(iter(train_loader)) 

f0, _ = seperate_f0_confidence(f0_with_confidence)

l_mod = cal_loudness_norm(l)

ae = TimbreFusionAE(timbre_emb_dim=256).to("cpu")
pt_file = f"{pt_file_dir}/decoder_v19_6_addmfft_energy_generator_best_9.pt"
ae.load_state_dict(torch.load(f"{pt_file}", map_location="cpu"))

synthsizer = HarmonicOscillator(is_smooth=True).to("cpu")
noise_filter = NoiseFilter().to("cpu")
enhance_synthesizer = EnhanceHarmonicOscillator(is_smooth=True).to("cpu")

harmonic_head_output, f0, noise_head_output, enhance_head_output = ae(s, s, l_mod, f0)
add = synthsizer(harmonic_head_output, f0)
sub = noise_filter(noise_head_output)
enhance = enhance_synthesizer(enhance_head_output, f0)
rec = add + sub  + enhance
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
enhance_l = extract_loudness(enhance.squeeze(dim=-1), A_weight)
add = add.view(-1).detach().numpy()
sub = sub.view(-1).detach().numpy()
enhance = enhance.view(-1).detach().numpy()
add_l = add_l.view(-1).detach().numpy()
sub_l = sub_l.view(-1).detach().numpy()
enhance_l = enhance_l.view(-1).detach().numpy()
add_l, sub_l = cal_loudness_norm(add_l), cal_loudness_norm(sub_l)
enhance_l = cal_loudness_norm(enhance_l)
l, rec_l = cal_loudness_norm(l), cal_loudness_norm(rec_l)

global_amp = global_amp.view(-1).detach().numpy()

def plot_result():
    p = plt.plot
    wf.write(f"{output_dir}/tmp/ori.wav", 16000, s)
    wf.write(f"{output_dir}/tmp/rec.wav", 16000, rec)
    wf.write(f"{output_dir}/tmp/add.wav", 16000, add)
    wf.write(f"{output_dir}/tmp/sub.wav", 16000, sub)
    plt.suptitle(fn[0])
    plt.subplot(431)
    p(s)
    plt.title("ori")
    plt.subplot(432)
    p(rec)
    plt.title("rec")
    plt.subplot(434)
    p(l)
    plt.title("ori_loudness")
    plt.subplot(433)
    p(global_amp)
    plt.title(f"global_amp")
    plt.subplot(435)
    p(rec_l)
    plt.title("rec_loudness")
    plt.subplot(436)
    diff_l = abs(l - rec_l)
    p(diff_l, color="red")
    plt.title(f"diff_loudness {diff_l.mean(): .3f}")
    plt.subplot(437)
    p(add_l)
    plt.title("Int Harm_loudness")
    plt.subplot(438)
    p(sub_l)
    plt.title("Noise_loudness")
    plt.subplot(439)
    p(enhance_l)
    plt.title("Non-Int Harm_loudness")
    plt.subplot(4, 3, 10)
    p(add)
    plt.title("Int Harm")
    plt.subplot(4, 3, 11)
    p(sub)
    plt.title("Noise")
    plt.subplot(4, 3, 12)
    p(enhance)
    plt.title("Non-Int Harm")
    plt.tight_layout()

plot_result()
#%%
out_dir = f"{output_dir}"
file_list_in_output_dir = glob(f"{out_dir}/*")
file_num = len(file_list_in_output_dir)
out_dir = f"{out_dir}/{file_num}"
os.makedirs(out_dir, exist_ok=True)
wf.write(f"{out_dir}/ori.wav", 16000, s)
wf.write(f"{out_dir}/rec.wav", 16000, rec)
wf.write(f"{out_dir}/Int_Harm.wav", 16000, add)
wf.write(f"{out_dir}/Noise.wav", 16000, sub)
wf.write(f"{out_dir}/Non-Int_Harm.wav", 16000, enhance)
plt.savefig(f"{out_dir}/result.png")