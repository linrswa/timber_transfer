# %%
import torch
from data.dataset import NSynthDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from glob import glob
import os 

from components.timbre_transformer.TimberTransformer import TimbreTransformer 
from components.timbre_transformer.utils import extract_loudness, get_A_weight, get_extract_pitch_needs, extract_pitch

use_mean_std = True
frequency_with_confidence = True

train_dataset = NSynthDataset(data_mode="train", sr=16000, frequency_with_confidence=frequency_with_confidence)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f0_confidence = next(iter(train_loader)) 

if frequency_with_confidence:
    f0, f0_confidence = f0_confidence[..., 0][...,: -1], f0_confidence[..., 1][...,: -1]

if use_mean_std:
    mean_loudness = -41.27331367041325
    std_loudness = 52.82343779478101552
    l_mod = (l - mean_loudness) / std_loudness

ddsp = TimbreTransformer(is_train=False, is_smooth=True, mlp_layer=3)
pt_file = "New_train_5_generator_30.pt"
ddsp.load_state_dict(torch.load(f"pt_file/{pt_file}"))
add, sub, rec, mu, logvar= ddsp(s, l_mod, f0)

f0_mask = f0_confidence < 0.85
f0[f0_mask] = torch.nan


A_weight = get_A_weight()
rec_l = extract_loudness(rec.squeeze(dim=-1), A_weight)
device, cr_model, m_sec = get_extract_pitch_needs(device="cpu")
rec_f0_confidence = extract_pitch(rec.squeeze(dim=-1), device=device, cr=cr_model, m_sec=m_sec, with_confidence=True)
rec_f0, rec_f0_confidence  = rec_f0_confidence[..., 0], rec_f0_confidence[..., 1]
rec_f0_mask = rec_f0_confidence < 0.85
rec_f0[rec_f0_mask] = torch.nan

# if use_mean_std:
#     rec_l = (rec_l.view(-1) - mean_loudness) / std_loudness

s = s.view(-1).numpy()
rec = rec.view(-1).detach().numpy()
rec_l = rec_l.view(-1).detach().numpy()
rec_f0 = rec_f0.view(-1).numpy()

def plot_result(s, rec, fn, rec_l, l):
    p = plt.plot
    wf.write("ori.wav", 16000, s)
    wf.write("rec.wav", 16000, rec)
    plt.suptitle(fn[0])
    plt.subplot(331)
    p(s)
    plt.title("ori")
    plt.subplot(332)
    p(rec)
    plt.title("rec")
    plt.subplot(334)
    p(l.view(-1))
    plt.title("ori_loudness")
    plt.subplot(335)
    p(rec_l)
    plt.title("rec_loudness")
    plt.subplot(336)
    p(abs(l.view(-1) - rec_l[:-1]), color="red")
    plt.title(f"diff_loudness {abs(l.view(-1) - rec_l[:-1]).mean(): .3f}")
    plt.subplot(337)
    p(f0.view(-1))
    plt.title("ori_f0")
    plt.subplot(338)
    p(rec_f0)
    plt.title("rec_f0")
    plt.subplot(339)
    p(abs(f0.view(-1) - rec_f0[:-1]), color="red")
    plt.title(f"diff_f0 {abs(f0.view(-1) - rec_f0[:-1]).nanmean(): .3f}")
    plt.tight_layout()

plot_result(s, rec, fn, rec_l, l)

#%%
out_dir = f"output/{pt_file}"
os.makedirs(out_dir, exist_ok=True)
file_list_in_output_dir = glob(f"{out_dir}/*")
file_num = len(file_list_in_output_dir)//3
file_name_with_dir = f"{out_dir}/{file_num}"
wf.write(f"{file_name_with_dir}_ori.wav", 16000, s)
wf.write(f"{file_name_with_dir}_rec.wav", 16000, rec)
plot_result(s, rec, fn, rec_l, l)
plt.savefig(f"{file_name_with_dir}.png")