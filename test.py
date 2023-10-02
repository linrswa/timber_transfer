# %%
import torch.nn as nn
from data.dataset import NSynthDataset
from torch.utils.data import DataLoader
from torchsummary import summary

from components.encoders import TimbreEncoder
from components.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from components.utils import discriminator_loss
from components.ddsp_modify.autoencoder import Encoder, Decoder
from components.ddsp_modify.ddsp import DDSP
from components.ddsp_modify.utils import extract_loudness, get_A_weight, mean_std_loudness


train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

ddsp = DDSP(is_train=False, is_smooth=True)
add, sub, rec, mu, logvar= ddsp(s, l, f)


def calculate_model_size(model: nn.Module):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'{model._get_name()} size: {size_all_mb:.3f}MB')


mpd = MultiPeriodDiscriminator()
mrd = MultiResolutionDiscriminator()

calculate_model_size(ddsp)
calculate_model_size(mpd)
calculate_model_size(mrd)

summary(ddsp, [(64000,), (250,), (250,)], device="cpu")
# summary(mpd, [(1, 64000), (1, 64000)], device="cpu")
# summary(mrd, [(64000,), (64000, )], device="cpu")


