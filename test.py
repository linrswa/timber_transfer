# %%
from dataset import NSynthDataset
from torch.utils.data import DataLoader
from components.encoders import SpeakerEncoder

train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
       
fn, s, l, f = next(iter(train_loader)) 

model = SpeakerEncoder()

mean_emb, convariance_emb = model(s)
