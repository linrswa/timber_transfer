# %%
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import sys
sys.path.append("..")
from tools.utils import seperate_f0_confidence

class NSynthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_mode: str,
        dir_path: str = "/dataset/NSynth/nsynth-subset",
        sr: int = 16000,
        frequency_with_confidence: bool = False,
    ):
        super().__init__()
        self.sr = sr
        self.dir_path = dir_path
        self.set_data_mode(data_mode)
        self.info_type = (
            {
                "signal": "signal",
                "loudness": "loudness",
                "frequency": "frequency_c",
            }
            if frequency_with_confidence
            else 
            {
                "signal": "signal",
                "loudness": "loudness",
                "frequency": "frequency",
            }
        )

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        signal_path = self.audio_list[idx]
        file_name = signal_path.split("/")[-1][:-4]
        signal = np.load(
            os.path.join(self.data_mode_dir_path, f"{self.info_type['signal']}/{file_name}.npy")
        ).astype("float32")
        loudness = np.load(
            os.path.join(self.data_mode_dir_path, f"{self.info_type['loudness']}/{file_name}.npy")
        ).astype("float32")[..., :-1]
        frequency = np.load(
            os.path.join(self.data_mode_dir_path, f"{self.info_type['frequency']}/{file_name}.npy")
        ).astype("float32")

        frequency_after = self.f0_distanglement_enhance(frequency)

        return (file_name, signal, loudness, frequency, frequency_after)

    def set_data_mode(self, data_mode: str):
        self.data_mode = data_mode
        self.data_mode_dir_path = f"{self.dir_path}/{data_mode}"
        signal_path = os.path.join(self.data_mode_dir_path, "signal/*")
        self.audio_list = glob(signal_path)

    def getitem_by_filename(self, fn: str):
        idx = self.audio_list.index(os.path.join(self.data_mode_dir_path, f"{self.info_type['signal']}/{fn}.npy"))
        return self.__getitem__(idx)
        
    def f0_distanglement_enhance(self, f0_with_confidence: np.ndarray):
        # only single file is supported
        f_w_c = f0_with_confidence.copy()
        f0, _ = f_w_c[..., 0][...,: -1], f_w_c[..., 1][...,: -1]
        non_zero_f0 = f0[f0 != 0]
        f0_mean = np.mean(non_zero_f0)
        f0_std = np.std(non_zero_f0)
        scale_mean = np.random.uniform(0.6, 1.5)
        scale_std = np.random.uniform(0.9, 1.2)

        # Adjust mean
        adjusted_f0 = non_zero_f0 * scale_mean
        # Adjust std
        adjusted_std = f0_std * scale_std
    
        adjusted_f0 = ((adjusted_f0 - f0_mean) / f0_std) * adjusted_std + f0_mean
        f0[f0 != 0] = adjusted_f0
        f_w_c[..., 0][...,: -1] = f0
        return f_w_c

USE_MEAN_STD = True
FREQUENCY_WITH_CONFIDENCE = True
USE_SMOOTH = True
output_dir = "../output"
pt_file_dir = "../pt_file"

train_dataset = NSynthDataset(data_mode="valid", sr=16000, frequency_with_confidence=FREQUENCY_WITH_CONFIDENCE)

train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
       
fn, s, l, f0_with_confidence, f0_after_with_confidence = next(iter(train_loader)) 

if FREQUENCY_WITH_CONFIDENCE:
    f0, _ = seperate_f0_confidence(f0_with_confidence)
    f0_after, _ = seperate_f0_confidence(f0_after_with_confidence)

s = s.view(-1).numpy()
f0 = f0.view(-1).numpy()
f0_after = f0_after.view(-1).numpy()

def plot_result():
    p = plt.plot
    plt.suptitle(fn[0])
    plt.subplot(411)
    p(s)
    plt.title("audio")
    plt.subplot(412)
    p(f0)
    plt.title("f0")
    plt.subplot(413)
    p(f0_after)
    plt.title("f0_fix")
    plt.tight_layout()
    plt.subplot(414)
    p(f0_after - f0)
    plt.title(f"{np.mean(np.abs(f0_after - f0))}")
    plt.tight_layout()

plot_result()