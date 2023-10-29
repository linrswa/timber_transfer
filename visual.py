# %%
import torch
from data.dataset import NSynthDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
import os 
from glob import glob

from components.ddsp_modify.ddsp import DDSP
from components.ddsp_modify.utils import extract_loudness, get_A_weight, mean_std_loudness

use_mean_std = True

train_dataset = NSynthDataset(data_mode="train", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

if use_mean_std:
    mean_loudness = -41.27331367041325
    std_loudness = 52.82343779478101552
    l = (l - mean_loudness) / std_loudness

ddsp = DDSP(is_train=False, is_smooth=True)
pt_file = "train 15_generator_best_6.pt"
ddsp.load_state_dict(torch.load(f"pt_file/{pt_file}"))
add, sub, rec, mu, logvar= ddsp(s, l, f)

A_weight = get_A_weight()
rec_l = extract_loudness(rec.squeeze(dim=-1), A_weight)

if use_mean_std:
    rec_l = (rec_l.view(-1) - mean_loudness) / std_loudness

s = s.view(-1).numpy()
rec = rec.view(-1).detach().numpy()
rec_l = rec_l.view(-1).detach().numpy()

def plot_result(s, rec, fn, rec_l, l):
    p = plt.plot
    wf.write("ori.wav", 16000, s)
    wf.write("rec.wav", 16000, rec)
    plt.suptitle(fn[0])
    plt.subplot(221)
    p(s)
    plt.title("ori")
    plt.subplot(222)
    p(rec)
    plt.title("rec")
    plt.subplot(223)
    p(l.view(-1))
    plt.title("ori_loudness")
    plt.subplot(224)
    p(rec_l)
    plt.title("rec_loudness")
    plt.tight_layout()

plot_result(s, rec, fn, rec_l, l)

#%%
out_dir = "output/train 15_generator_best_6"
os.makedirs(out_dir, exist_ok=True)
file_list_in_output_dir = glob(f"{out_dir}/*")
file_num = len(file_list_in_output_dir)//3
file_name_with_dir = f"{out_dir}/{file_num}"
wf.write(f"{file_name_with_dir}_ori.wav", 16000, s)
wf.write(f"{file_name_with_dir}_rec.wav", 16000, rec)
plot_result(s, rec, fn, rec_l, l)
plt.savefig(f"{file_name_with_dir}.png")