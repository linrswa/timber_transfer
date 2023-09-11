import torch
from torch.utils.data import DataLoader
import tqdm

from components.ddsp_modify.ddsp import DDSP
from dataset import NSynthDataset


train_dataset = NSynthDataset(data_mode="test", sr=16000)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
model = DDSP()


for fn, s, l ,f in tqdm(train_dataset):
    pass

