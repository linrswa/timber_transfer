# %%
from dataset import NSynthDataset
from torch.utils.data import DataLoader
from components.encoders import SpeakerEncoder
from components.discriminators import MulitiPeriodDiscriminator, MultiResolutionDiscriminator

train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

# model = MulitiPeriodDiscriminator()
model = MultiResolutionDiscriminator()

y_d_rs, y_d_gs, fmap_rs, fmap_gs = model(s.unsqueeze(dim=1), s.unsqueeze(dim=1))