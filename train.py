#%%
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import json

from components.ddsp_modify.ddsp import DDSP
from components.discriminators import MultiResolutionDiscriminator, MultiPeriodDiscriminator
from components.utils import generator_loss, discriminator_loss, feature_loss
from utils import mel_spectrogram
from dataset import NSynthDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open("./config.json") as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
generator = DDSP().to(device)
mrd = MultiResolutionDiscriminator().to(device)
mpd = MultiPeriodDiscriminator().to(device)

optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

last_epoch = -1
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

generator.train()
mrd.train()
mpd.train()


num_epochs = 100
for epoch in range(num_epochs):
    for fn, s, l ,f in tqdm(train_loader):

        s = s.to(device)
        l = l.to(device)    
        f = f.to(device)
        
        add, sub, y_g_hat = generator(s, l, f)
        y_mel = mel_spectrogram(s, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, center=False)
        y_g_hat_mel = mel_spectrogram(y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, center=False)
        
        s = s.unsqueeze(dim=1)
        y_g_hat = y_g_hat.permute(0, 2, 1).contiguous()
        # Train Discriminator
        optim_d.zero_grad()
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(s, y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_dr_hat_r, y_dr_hat_g, _, _ = mrd(s, y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_dr_hat_r, y_dr_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        
        loss_disc_all.backward()
        optim_d.step()
        
        # Train Generator
        optim_g.zero_grad()
       
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(s, y_g_hat)
        y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(s, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_r_r, fmap_r_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_r, losses_gen_r = generator_loss(y_dr_hat_g)
        loss_gen_all = loss_gen_r + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        optim_g.step() 
        
        print(f"loss_fm_f: {loss_fm_f}, loss_fm_s: {loss_fm_s}, loss_gen_f: {loss_gen_f}, loss_gen_r: {loss_gen_r}, loss_mel: {loss_mel}")
        print(f"loss_disc_all: {loss_disc_all}, loss_gen_all: {loss_gen_all}")
        
        scheduler_g.step()
        scheduler_d.step()


