
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from data.dataset import NSynthDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import inquirer
from glob import glob

from utils import cal_mean_std_loudness
from components.timbre_transformer.TimberTransformer import TimbreTransformer
from components.timbre_transformer.utils import extract_loudness, get_A_weight, mean_std_loudness


def get_loudness_l1_loss(
    signal: Tensor,
    target_l: Tensor,
    aw: Tensor,
    mean_std_dict: dict,
    norm: bool = False
    ):
    signal = signal.view(signal.shape[0], -1)
    y_l = extract_loudness(signal, aw)[..., :-1]

    if norm:
        target_l = cal_mean_std_loudness(target_l, mean_std_dict)
        y_l = cal_mean_std_loudness(y_l, mean_std_dict)

    l_l1_loss = F.l1_loss(y_l, target_l)
    return l_l1_loss

@torch.no_grad()
def valid_model_loudness(
    model: nn.Module, data_mode: str, batch: int
):
    mean_std_dict = {}
    dataset = NSynthDataset(data_mode=data_mode, sr=16000)
    valid_loader = DataLoader(dataset, batch_size=batch, num_workers=8)
    mean_std_dict["mean_loudness"], mean_std_dict["std_loudness"]= mean_std_loudness(valid_loader)
    device = next(model.parameters()).device
    loss_sum = 0
    num_samples = 0
    aw = get_A_weight().to(device)
    for val_p, val_s, val_l, val_f in tqdm(valid_loader):
        val_s, val_l, val_f = val_s.to(device), val_l.to(device), val_f.to(device)
        out_add, out_sub, out_rec, out_mu, out_logvar = model(val_s, val_l, val_f)
        loss = get_loudness_l1_loss(out_rec, val_l, aw, mean_std_dict,  norm=True)
        loss_sum += loss.item() * val_s.size(0)
        num_samples += val_s.size(0)
    mean_loss = loss_sum / num_samples
    return mean_loss

pt_list_list = glob("./pt_file/*generator*.pt")
pt_list_list = sorted(pt_list_list)
pt_fonfirm = {
    inquirer.List("pt_file", message="Choose a pt file", choices=pt_list_list)
}
pt_file = inquirer.prompt(pt_fonfirm)["pt_file"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = TimbreTransformer(is_smooth=True, mlp_layer=3, n_harms=101).to(device)
generator.load_state_dict(torch.load(pt_file))

data_mode = "valid"
l1_loudness = valid_model_loudness(generator, data_mode, 32)
print(f"{data_mode}: {pt_file.split('/')[-1]}: {l1_loudness}", file=open("l1_loudness.txt", "a"))
