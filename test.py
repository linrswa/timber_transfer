# %%
import torch
from dataset import NSynthDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from components.encoders import TimbreEncoder
from components.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from components.utils import discriminator_loss
from components.ddsp_modify.autoencoder import Encoder, Decoder
from components.ddsp_modify.ddsp import DDSP


train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

ddsp = DDSP(is_train=False)
ddsp.load_state_dict(torch.load("./pt_file/test5_generator_58.pt"))
add, sub, rec, mu, logvar= ddsp(s, l, f)

import scipy.io.wavfile as wf
import matplotlib.pyplot as plt 

s = s.view(-1).numpy()
rec = rec.view(-1).detach().numpy()
print(fn)
wf.write("ori.wav", 16000, s)
wf.write("rec.wav", 16000, rec)
plt.subplot(211)
plt.plot(s)
plt.title("ori")
plt.subplot(212)
plt.plot(rec)
plt.title("rec")
plt.tight_layout()
#%%
# model = MultiPeriodDiscriminator()
# y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(s.unsqueeze(dim=1), s.unsqueeze(dim=1))
# loss, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)