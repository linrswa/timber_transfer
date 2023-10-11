# %%
import torch
from data.dataset import NSynthDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

from components.encoders import TimbreEncoder
from components.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from components.utils import discriminator_loss
from components.ddsp_modify.autoencoder import Encoder, Decoder
from components.ddsp_modify.ddsp import DDSP
from components.ddsp_modify.utils import extract_loudness, get_A_weight, mean_std_loudness

use_mean_std = False

train_dataset = NSynthDataset(data_mode="train", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f, mfcc = next(iter(train_loader)) 

if use_mean_std:
    mean_loudness, std_loudness = mean_std_loudness(train_loader)
    l = (l - mean_loudness) / std_loudness

ddsp = DDSP(is_train=False, use_extract_mfcc=False)
ddsp.load_state_dict(torch.load("./pt_file/train 10_generator_60.pt"))
add, sub, rec, mu, logvar= ddsp(mfcc, l, f)

A_weight = get_A_weight()
rec_l = extract_loudness(rec.squeeze(dim=-1), A_weight)

if use_mean_std:
    rec_l = (rec_l.view(-1) - mean_loudness) / std_loudness

s = s.view(-1).numpy()
rec = rec.view(-1).detach().numpy()
rec_l = rec_l.view(-1).detach().numpy()
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
#%%
# model = MultiPeriodDiscriminator()
# y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(s.unsqueeze(dim=1), s.unsqueeze(dim=1))
# loss, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)