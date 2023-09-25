# %%
import torch
from dataset import NSynthDataset
from torch.utils.data import DataLoader

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
# ddsp.load_state_dict(torch.load("./pt_file/train2_generator_7.pt"))
add, sub, rec, mu, logvar= ddsp(s, l, f)