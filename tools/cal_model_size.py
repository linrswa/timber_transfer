# %%
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary

import sys
sys.path.append(".")
from data.dataset import NSynthDataset
from components.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from components.timbre_transformer.TimberTransformer import TimbreTransformer
from ddsp_ori.ddsp import DDSP as DDSP_origin


train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

timbre_transformer = TimbreTransformer(is_train=False, is_smooth=True, n_harms=101, timbre_emb_dim=128)

add, sub, rec, mu, logvar, global_amp = timbre_transformer(s, l, f)


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
ddsp_origin = DDSP_origin()

calculate_model_size(ddsp_origin)
calculate_model_size(timbre_transformer)
calculate_model_size(mpd)
calculate_model_size(mrd)


# summary(ddsp_origin, [s.shape, l.shape, f.shape], device="cpu")

summary(timbre_transformer, input_data=(s, l, f), device="cpu")
# summary(mpd, [s.shape, s.shape], device="cpu")
# summary(mrd, [s.shape, s.shape], device="cpu")


