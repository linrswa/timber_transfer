#%%
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from components.timbre_transformer.TimberTransformer import TimbreTransformer
from components.discriminators import MultiResolutionDiscriminator, MultiPeriodDiscriminator
from components.utils import generator_loss, discriminator_loss, feature_loss, kl_loss
from components.timbre_transformer.utils import extract_loudness, get_A_weight
from tools.utils import mel_spectrogram, get_hyparam, get_mean_std_dict, cal_mean_std_loudness
from tools.utils import multiscale_fft, safe_log
from data.dataset import NSynthDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
run_name = "train11"
notes = "Add InputAttBlock to get more information from f0 and loudness, change out_gru_f0 for only for later concate."

h = get_hyparam()

def cal_mean_loss(total_mean_loss, batch_mean_loss, n_element):
    return (batch_mean_loss.item() - total_mean_loss) / n_element

mean_std_dict = get_mean_std_dict("train", 128)

train_dataset = NSynthDataset(data_mode="train", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=8 , num_workers=4, shuffle=True)
generator = TimbreTransformer(is_train=True, is_smooth=True, mlp_layer=h.mlp_layer).to(device)
mrd = MultiResolutionDiscriminator().to(device)

optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
optim_d = torch.optim.AdamW(mrd.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

last_epoch = -1
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

generator.train()
mrd.train()

config = {
    "loss_weight": h.loss_weight,
    "mlp_layer": h.mlp_layer,
}

wandb.init(
    project="TimbreTransformer",
    name=run_name, 
    notes=notes,
    config=config
    )


num_epochs = 300
# set init value for logging
best_loss = float("inf")
step = 0
n_element = 0

total_mean_disc_loss = {
    "disc_r": 0,
    "disc_all": 0,
}    


total_mean_gen_loss = {
    "gen_r": 0,
    "gen_fm_r": 0,
    "gen_mel": 0,
    "gen_multiscale_fft": 0,
    "gen_kl": 0,
    "gen_loudness": 0,
    "gen_all": 0,
}

step_loss_50 = {
    "gen_r": 0,
    "gen_fm_r": 0,
    "gen_mel": 0,
    "gen_multiscale_fft": 0,
    "gen_kl": 0,
    "gen_loudness": 0,
    "gen_all": 0,
    "disc_r": 0,
    "disc_all": 0,
}


A_weight = get_A_weight().to(device)
for epoch in tqdm(range(num_epochs)):
    for fn, s, l ,f in tqdm(train_loader):
        
        s = s.to(device)
        l = l.to(device)    
        f = f.to(device)

        l_norm = cal_mean_std_loudness(l, mean_std_dict)
        
        add, sub, y_g_hat, mu, logvar, global_amp = generator(s, l_norm, f)

        y_mel = mel_spectrogram(s, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, center=False)
        y_g_hat_mel = mel_spectrogram(y_g_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, center=False)
        
        s = s.unsqueeze(dim=1)
        y_g_hat = y_g_hat.permute(0, 2, 1).contiguous()
        # Train Discriminator
        optim_d.zero_grad()

        # MRD
        y_dr_hat_r, y_dr_hat_g, _, _ = mrd(s, y_g_hat.detach())
        loss_disc_r, losses_disc_r_r, losses_disc_s_g = discriminator_loss(y_dr_hat_r, y_dr_hat_g)

        loss_disc_all = loss_disc_r 
        
        loss_disc_all.backward()
        optim_d.step()
        
        # Train Generator
        optim_g.zero_grad()
       
        # Additional loudness loss
        rec_l = extract_loudness(y_g_hat.squeeze(dim=1), A_weight)[:, :-1]
        rec_l = cal_mean_std_loudness(rec_l, mean_std_dict)
        loss_gen_loudness = F.l1_loss(rec_l, l_norm) * h.loss_weight["gen_loudness"] 

        # Multiscale FFT loss
        ori_stft = multiscale_fft(s.squeeze(dim=1))
        rec_stft = multiscale_fft(y_g_hat.squeeze(dim=1))
        loss_gen_multiscale_fft = 0 
        for s_x, s_y in zip(ori_stft, rec_stft):
            linear_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss_gen_multiscale_fft += linear_loss + log_loss

        loss_gen_multiscale_fft *= h.loss_weight["gen_multiscale_fft"]

        loss_gen_mel = F.l1_loss(y_mel, y_g_hat_mel) * h.loss_weight["gen_mel"]
        loss_gen_kl = kl_loss(mu, logvar) * h.loss_weight["gen_kl"]

        y_dr_hat_r, y_dr_hat_g, fmap_r_r, fmap_r_g = mrd(s, y_g_hat)
        loss_gen_fm_r = feature_loss(fmap_r_r, fmap_r_g)
        loss_gen_r, losses_gen_r = generator_loss(y_dr_hat_g)
        loss_gen_all = loss_gen_r + loss_gen_fm_r + loss_gen_mel + loss_gen_kl + loss_gen_multiscale_fft 
        

        loss_gen_all.backward()
        optim_g.step() 
        
        scheduler_g.step()
        scheduler_d.step()

        # calculate mean loss for logging'
        step += 1
        n_element += 1
        # disc
        for k, v in total_mean_disc_loss.items():
            total_mean_disc_loss[k] += cal_mean_loss(v, locals()[f"loss_{k}"], n_element)
        
        # gen
        for k, v in total_mean_gen_loss.items():
            total_mean_gen_loss[k] += cal_mean_loss(v, locals()[f"loss_{k}"], n_element)

        
        # logging
        log_step_loss_50 = {}
        if step % 50 == 0:
            for k, _ in step_loss_50.items():
                log_step_loss_50[f"50step_loss_{k}"] = locals()[f"loss_{k}"].item()
            wandb.log(log_step_loss_50)


    epoch_loss = {}
    for k, v in total_mean_disc_loss.items():
        epoch_loss[f"epoch_loss_{k}"] = v
    for k, v in total_mean_gen_loss.items():
        epoch_loss[f"epoch_loss_{k}"] = v
    wandb.log(epoch_loss)


    print(total_mean_gen_loss)

    print(f"loss_disc_all: {total_mean_disc_loss['disc_all']}, loss_gen_all: {total_mean_gen_loss['gen_all']}")

    if total_mean_gen_loss["gen_all"] < best_loss:
        best_loss = total_mean_gen_loss["gen_all"]
        torch.save(generator.state_dict(), f"./pt_file/{run_name}_generator_best_{epoch}.pt")
        torch.save(mrd.state_dict(), f"./pt_file/{run_name}_mrd_best_{epoch}.pt")
        print(f"save best model at epoch {epoch}")
    elif epoch % 10 == 0:
        torch.save(generator.state_dict(), f"./pt_file/{run_name}_generator_{epoch}.pt")
        torch.save(mrd.state_dict(), f"./pt_file/{run_name}_mrd_{epoch}.pt")   
        print(f"save model at epoch {epoch}")

    # reset value for logging
    n_element = 0
    for k, v in total_mean_disc_loss.items():
        total_mean_disc_loss[k] = 0

    for k, v in total_mean_gen_loss.items():
        total_mean_gen_loss[k] = 0
# %%