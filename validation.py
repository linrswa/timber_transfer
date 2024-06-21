# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import NSynthDataset
from tools.utils import cal_mean_std_loudness, mask_f0_with_confidence, seperate_f0_confidence
from components.timbre_transformer.TimberTransformer import TimbreTransformer
from components.timbre_transformer.utils import extract_loudness, get_A_weight, mean_std_loudness
from components.timbre_transformer.utils import extract_pitch, get_extract_pitch_needs

def get_loudness_l1_loss(
    rec_signal: Tensor,
    target_l: Tensor,
    aw: Tensor,
    mean_std_dict: dict,
    norm: bool = False,
):
    rec_signal = rec_signal.view(rec_signal.shape[0], -1)
    y_l = extract_loudness(rec_signal, aw)[..., :-1]

    if norm:
        target_l = cal_mean_std_loudness(target_l, mean_std_dict)
        y_l = cal_mean_std_loudness(y_l, mean_std_dict)
        
    l_l1_loss = F.l1_loss(y_l, target_l)
    return l_l1_loss

def get_pitch_l1_loss(
    signal: Tensor,
    target_f0_c: Tensor,
    device: torch.device,
    cr: int,
    m_sec: int,
):
    signal = signal.view(signal.shape[0], -1)
    y_f0_c = extract_pitch(signal, device, cr, m_sec, with_confidence=True)
    target_f0 = mask_f0_with_confidence(target_f0_c, threshold=0.85, return_midi=True)
    y_f0 = mask_f0_with_confidence(y_f0_c, threshold=0.85, return_midi=True)
    y_f0 = y_f0.to(device)
    f_l1_loss = abs(target_f0 - y_f0).nanmean() 
    return f_l1_loss
    

@torch.no_grad()
def valid_model(
    model: nn.Module, data_mode: str, batch: int
):
    mean_std_dict = {}
    dataset = NSynthDataset(data_mode=data_mode, sr=16000, frequency_with_confidence=True)
    valid_loader = DataLoader(dataset, batch_size=batch, num_workers=8)
    mean_std_dict["mean_loudness"], mean_std_dict["std_loudness"]= mean_std_loudness(valid_loader)
    device = next(model.parameters()).device

    loss_l_sum = 0
    loss_f_sum = 0
    num_samples = 0

    aw = get_A_weight().to(device)
    device, cr, m_sec = get_extract_pitch_needs(device)
    for val_p, val_s, val_l, val_f0_c in tqdm(valid_loader):
        val_s, val_l, val_f0_c = val_s.to(device), val_l.to(device), val_f0_c.to(device)
        val_f0, val_f0_confidence = seperate_f0_confidence(val_f0_c)
        norm_l = cal_mean_std_loudness(val_l, mean_std_dict)
        out_add, out_sub, out_rec, out_mu, out_logvar, global_amp = model(val_s, norm_l, val_f0)

        loss_l = get_loudness_l1_loss(out_rec, val_l, aw, mean_std_dict, norm=True)
        loss_f = get_pitch_l1_loss(out_rec, val_f0_c, device, cr, m_sec)
        loss_l_sum += loss_l.item() * val_s.size(0)
        loss_f_sum += loss_f.item() * val_s.size(0)

        num_samples += val_s.size(0)

    mean_l_loss = loss_l_sum / num_samples
    mean_f_loss = loss_f_sum / num_samples

    return mean_l_loss, mean_f_loss

if __name__ == "__main__":
    # pt_list_list = glob("./pt_file/*generator*.pt")
    # pt_list_list = sorted(pt_list_list)
    # pt_fonfirm = {
    #     inquirer.List("pt_file", message="Choose a pt file", choices=pt_list_list)
    # }
    # pt_file = inquirer.prompt(pt_fonfirm)["pt_file"]
    pt_file = "./pt_file/decoder_v15_9(mfcc)_generator_best_8.pt"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    generator = TimbreTransformer(is_smooth=True, n_harms=101, timbre_emb_dim=256).to(device)
    generator.load_state_dict(torch.load(pt_file))

    data_mode = "valid"
    l1_loudness, l1_f0 = valid_model(generator, data_mode, 16)

    print(f""""
        finish {pt_file.split('/')[-1]}\n 
        \t loudness loss: {l1_loudness}\n
        \t pitch loss: {l1_f0}\n
        """ )

    with open("validation_log.txt", "a") as f:
        f.write(f"{data_mode}: {pt_file.split('/')[-1]}\n")
        f.write(f"\tloudness loss: {l1_loudness}\n")
        f.write(f"\tpitch loss: {l1_f0}\n")
