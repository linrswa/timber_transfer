# %%
from dataset import NSynthDataset
from torch.utils.data import DataLoader
import torch

from components.encoders import TimbreEncoder
from components.discriminators import MulitiPeriodDiscriminator, MultiResolutionDiscriminator
from components.utils import discriminator_loss
from components.ddsp_modify.autoencoder import Encoder, Decoder
from components.ddsp_modify.ddsp import DDSP


train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

ddsp = DDSP()
add, sub, rec = ddsp(s, l, f)
#%%
# model = MulitiPeriodDiscriminator()
# y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(s.unsqueeze(dim=1), s.unsqueeze(dim=1))
# loss, r_losses, g_losses = discriminator_loss(y_d_rs, y_d_gs)