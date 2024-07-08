#%%
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from components.timbre_transformer.TimberTransformer import TimbreTransformer
from components.discriminators import MultiPeriodDiscriminator
from components.timbre_transformer.utils import extract_loudness, get_A_weight
from components.timbre_transformer.encoder import EngryEncoder
from tools.utils import mel_spectrogram, get_hyparam, get_mean_std_dict, cal_mean_std_loudness
from tools.loss_collector import LossCollector as L
from data.dataset import NSynthDataset

#MARK: Train setting
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
run_name = "decoder_v21_3_addmfftx2_energy"
notes = "mfft * 2"
batch_size = 16

h = get_hyparam()

def cal_mean_loss(total_mean_loss, batch_mean_loss, n_element):
    return (batch_mean_loss.item() - total_mean_loss) / n_element

mean_std_dict = get_mean_std_dict("train", 128)

train_dataset = NSynthDataset(data_mode="train", sr=16000, with_f0_distanglement=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
generator = TimbreTransformer(is_train=True, is_smooth=True, timbre_emb_dim=256).to(device)
mpd = MultiPeriodDiscriminator().to(device)

# optim_g = torch.optim.Adam(generator.parameters(), 0.001)
optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
optim_d = torch.optim.AdamW(mpd.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

last_epoch = -1
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

generator.train()
mpd.train()

config = {
    "loss_weight": h.loss_weight,
    "mlp_layer": h.mlp_layer,
}

wandb.init(
    project="TimbreTransformer_v4",
    name=run_name, 
    notes=notes,
    config=config
    )


#MAKR: Set init value for logging
num_epochs = 300
best_loss = float("inf")
step = 0
n_element = 0

init_loss = {
    "gen_period": 0,
    "gen_fm_period": 0,
    "gen_mel": 0,
    "gen_multiscale_fft": 0,
    "gen_kl": 0,
    "gen_loudness": 0,
    "gen_all": 0,
    "disc_period": 0,
    "disc_all": 0,
}

total_mean_loss = init_loss

step_loss_50 = init_loss

A_weight = get_A_weight().to(device)
#MARK: Train loop
for epoch in tqdm(range(num_epochs)):
    for fn, s, l ,f in tqdm(train_loader):
        
        s = s.to(device)
        l = l.to(device)    
        f = f.to(device)

        l_norm = cal_mean_std_loudness(l, mean_std_dict)
        
        add, sub, y_g_hat, mu, logvar, global_amp = generator(s, l_norm, f, s)

        y_mel = mel_spectrogram(s, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, center=False)
        y_g_hat_mel = mel_spectrogram(y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, center=False)
        
        s = s.unsqueeze(dim=1)
        y_g_hat = y_g_hat.permute(0, 2, 1).contiguous()

        #MARK: Train Discriminator
        optim_d.zero_grad()
        y_mpd_hat_r, y_mpd_hat_g, _, _ = mpd(s, y_g_hat.detach())
        loss_disc_period = L.discriminator_loss(y_mpd_hat_r, y_mpd_hat_g)

        loss_disc_all = loss_disc_period
        loss_disc_all.backward()
        optim_d.step()
        
        #MARK: Train Generator
        optim_g.zero_grad()
        loss_gen_multiscale_fft = L.multiscale_fft_loss(s, y_g_hat, reduction='mean') * h.loss_weight["gen_multiscale_fft"]
        loss_gen_mel = F.l1_loss(y_mel, y_g_hat_mel, reduction='mean') * h.loss_weight["gen_mel"]
        loss_gen_kl = L.kl_loss(mu, logvar) * h.loss_weight["gen_kl"]
        _, y_mpd_hat_g, fmap_mpd_r, fmap_mpd_g = mpd(s, y_g_hat)
        loss_gen_fm_period = L.feature_loss(fmap_mpd_r, fmap_mpd_g) * h.loss_weight["gen_fm_period"]
        loss_gen_period= L.generator_loss(y_mpd_hat_g)
        loss_gen_all = loss_gen_period + loss_gen_fm_period + loss_gen_mel + loss_gen_kl + loss_gen_multiscale_fft
        loss_gen_all.backward()
        optim_g.step() 

        #MARK: calculate mean loss for logging not in tranning
        with torch.no_grad():
            rec_l = extract_loudness(y_g_hat.squeeze(dim=1), A_weight)[:, :-1]
            rec_l = cal_mean_std_loudness(rec_l, mean_std_dict)
            loss_gen_loudness = F.l1_loss(rec_l, l_norm) 

        step += 1
        n_element += 1

       #MARK: update scheduler 
        scheduler_g.step()
        scheduler_d.step()
        #lr update from ddsp
        # if step % 10000 == 0:
        #     optim_g.param_groups[0]["lr"] *= 0.98
        #     print(f"update lr to {optim_g.param_groups[0]['lr']}")
            
        for k, v in total_mean_loss.items():
            total_mean_loss[k] += cal_mean_loss(v, locals()[f"loss_{k}"], n_element)

        log_step_loss_50 = {}
        if step % 50 == 0:
            for k, _ in step_loss_50.items():
                log_step_loss_50[f"50step_loss_{k}"] = locals()[f"loss_{k}"].item()
            wandb.log(log_step_loss_50)

    epoch_loss = {}
    for k, v in total_mean_loss.items():
        epoch_loss[f"epoch_loss_{k}"] = v
    wandb.log(epoch_loss)

    print(total_mean_loss)

    if total_mean_loss["gen_all"] < best_loss:
        best_loss = total_mean_loss["gen_all"]
        torch.save(generator.state_dict(), f"./pt_file/{run_name}_generator_best_{epoch}.pt")
        # torch.save(mpd.state_dict(), f"./pt_file/{run_name}_mrd_best_{epoch}.pt")
        print(f"save best model at epoch {epoch}")
    elif epoch % 10 == 0:
        torch.save(generator.state_dict(), f"./pt_file/{run_name}_generator_{epoch}.pt")
        # torch.save(mpd.state_dict(), f"./pt_file/{run_name}_mrd_{epoch}.pt")   
        print(f"save model at epoch {epoch}")

    #MARK: reset value for logging
    n_element = 0
    for k, v in total_mean_loss.items():
        total_mean_loss[k] = 0