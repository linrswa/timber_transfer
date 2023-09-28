#%%
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import json
import wandb

from components.ddsp_modify.ddsp import DDSP
from components.ddsp_modify.utils import mean_std_loudness
from components.discriminators import MultiResolutionDiscriminator, MultiPeriodDiscriminator
from components.utils import generator_loss, discriminator_loss, feature_loss, kl_loss
from components.ddsp_modify.utils import extract_loudness, get_A_weight
from utils import mel_spectrogram
from dataset import NSynthDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open("./config.json") as f:
    data = f.read()

json_config = json.loads(data)
h = AttrDict(json_config)

def cal_mean_loss(total_mean_loss, batch_mean_loss, n_element):
    return (batch_mean_loss.item() - total_mean_loss) / n_element
    

train_dataset = NSynthDataset(data_mode="train", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=8 , num_workers=4, shuffle=True)
generator = DDSP(is_train=True).to(device)
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


run_name = "train7"
wandb.init(project="ddsp_modify", name=run_name, tags="add loudness loss")

num_epochs = 300
# set init value for logging
step = 0
best_loss = float("inf")

n_element = 0
total_mean_loss_disc_f = 0
total_mean_loss_disc_r = 0
total_mean_loss_disc_all = 0

total_mean_loss_gen_f = 0
total_mean_loss_gen_loudness = 0
total_mean_loss_gen_r = 0
total_mean_loss_gen_fm_f = 0
total_mean_loss_gen_fm_r = 0
total_mean_loss_gen_mel = 0
total_mean_loss_gen_kl = 0
total_mean_loss_gen_all = 0

A_weight = get_A_weight().to(device)
for epoch in tqdm(range(num_epochs)):
    for fn, s, l ,f in tqdm(train_loader):
        
        s = s.to(device)
        l = l.to(device)    
        f = f.to(device)
        
        
        add, sub, y_g_hat, mu, logvar = generator(s, l, f)
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
        loss_disc_r, losses_disc_r_r, losses_disc_s_g = discriminator_loss(y_dr_hat_r, y_dr_hat_g)

        loss_disc_all = loss_disc_r + loss_disc_f
        
        loss_disc_all.backward()
        optim_d.step()
        
        # Train Generator
        optim_g.zero_grad()
       
        # Additional loudness loss
        rec_l = extract_loudness(y_g_hat.squeeze(dim=1), A_weight)
        loss_gen_loudness = F.l1_loss(rec_l[:, :-1], l) * 0.1

        loss_gen_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

        loss_gen_kl = kl_loss(mu, logvar) * 0.01

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(s, y_g_hat)
        y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(s, y_g_hat)
        loss_gen_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_fm_r = feature_loss(fmap_r_r, fmap_r_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_r, losses_gen_r = generator_loss(y_dr_hat_g)
        loss_gen_all = loss_gen_r + loss_gen_f + loss_gen_fm_r + loss_gen_fm_f + loss_gen_mel + loss_gen_kl + loss_gen_loudness

        loss_gen_all.backward()
        optim_g.step() 
        
        scheduler_g.step()
        scheduler_d.step()

        # calculate mean loss for logging'
        step += 1
        n_element += 1
        # disc
        total_mean_loss_disc_f += cal_mean_loss(total_mean_loss_disc_f, loss_disc_f, n_element)
        total_mean_loss_disc_r += cal_mean_loss(total_mean_loss_disc_r, loss_disc_r, n_element)
        total_mean_loss_disc_all += cal_mean_loss(total_mean_loss_disc_all, loss_disc_all, n_element)

        # gen
        total_mean_loss_gen_loudness += cal_mean_loss(total_mean_loss_gen_loudness, loss_gen_loudness, n_element)
        total_mean_loss_gen_f += cal_mean_loss(total_mean_loss_gen_f, loss_gen_f, n_element)
        total_mean_loss_gen_r += cal_mean_loss(total_mean_loss_gen_r, loss_gen_r, n_element)
        total_mean_loss_gen_fm_f += cal_mean_loss(total_mean_loss_gen_fm_f, loss_gen_fm_f, n_element)
        total_mean_loss_gen_fm_r += cal_mean_loss(total_mean_loss_gen_fm_r, loss_gen_fm_r, n_element)
        total_mean_loss_gen_mel += cal_mean_loss(total_mean_loss_gen_mel, loss_gen_mel, n_element)
        total_mean_loss_gen_kl += cal_mean_loss(total_mean_loss_gen_kl, loss_gen_kl, n_element) 
        total_mean_loss_gen_all += cal_mean_loss(total_mean_loss_gen_all, loss_gen_all, n_element)

        # logging
        if step % 50 == 0:
            wandb.log(
                {
                    "50step_loss_disc_f": total_mean_loss_disc_f,
                    "50step_loss_disc_r": total_mean_loss_disc_r,
                    "50step_loss_disc_all": total_mean_loss_disc_all,
                    "50step_loss_gen_loudness": total_mean_loss_gen_loudness,
                    "50step_loss_gen_f": total_mean_loss_gen_f,
                    "50step_loss_gen_r": total_mean_loss_gen_r,
                    "50step_loss_gen_fm_f": total_mean_loss_gen_fm_f,
                    "50step_loss_gen_fm_r": total_mean_loss_gen_fm_r,
                    "50step_loss_gen_mel": total_mean_loss_gen_mel,
                    "50step_loss_gen_kl": total_mean_loss_gen_kl,
                    "50step_loss_gen_all": total_mean_loss_gen_all,
                }
            )


    wandb.log(
        {
            "epoch_loss_disc_f": total_mean_loss_disc_f,
            "epoch_loss_disc_r": total_mean_loss_disc_r,
            "epoch_loss_disc_all": total_mean_loss_disc_all,
            "epoch_loss_gen_loudness": total_mean_loss_gen_loudness,
            "epoch_loss_gen_f": total_mean_loss_gen_f,
            "epoch_loss_gen_r": total_mean_loss_gen_r,
            "epoch_loss_gen_fm_f": total_mean_loss_gen_fm_f,
            "epoch_loss_gen_fm_r": total_mean_loss_gen_fm_r,
            "epoch_loss_gen_mel": total_mean_loss_gen_mel,
            "epoch_loss_gen_kl": total_mean_loss_gen_kl,
            "epoch_loss_gen_all": total_mean_loss_gen_all,
        }
    )


    print(
            f"loss_fm_f: {total_mean_loss_gen_fm_f}, loss_fm_s: {total_mean_loss_gen_fm_r}, \
            loss_gen_f: {total_mean_loss_gen_f}, loss_gen_r: {total_mean_loss_gen_r}, \
            loss_mel: {total_mean_loss_gen_mel}, loss_kl: {total_mean_loss_gen_kl} \
            loss_loudness: {total_mean_loss_gen_loudness}"
        )
    print(f"loss_disc_all: {total_mean_loss_disc_all}, loss_gen_all: {total_mean_loss_gen_all}")

    if total_mean_loss_gen_all < best_loss:
        best_loss = total_mean_loss_gen_all
        torch.save(generator.state_dict(), f"./pt_file/{run_name}_generator_best_{epoch}.pt")
        torch.save(mrd.state_dict(), f"./pt_file/{run_name}_mrd_best_{epoch}.pt")
        torch.save(mpd.state_dict(), f"./pt_file/{run_name}_mpd_best_{epoch}.pt")
        print(f"save best model at epoch {epoch}")
    elif epoch % 10 == 0:
        torch.save(generator.state_dict(), f"./pt_file/{run_name}_generator_{epoch}.pt")
        torch.save(mrd.state_dict(), f"./pt_file/{run_name}_mrd_{epoch}.pt")   
        torch.save(mpd.state_dict(), f"./pt_file/{run_name}_mpd_{epoch}.pt")
        print(f"save model at epoch {epoch}")

    # reset value for logging
    n_element = 0
    total_mean_loss_disc_f = 0
    total_mean_loss_disc_r = 0
    total_mean_loss_disc_all = 0

    total_mean_loss_gen_loudness = 0
    total_mean_loss_gen_f = 0
    total_mean_loss_gen_r = 0
    total_mean_loss_gen_fm_f = 0
    total_mean_loss_gen_fm_r = 0
    total_mean_loss_gen_mel = 0
    total_mean_loss_gen_kl = 0
    total_mean_loss_gen_all = 0