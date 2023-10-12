import torch
import os
import numpy as np
from glob import glob

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
        self.dir_path = f"{dir_path}/{data_mode}"
        signal_path = os.path.join(self.dir_path, "signal/*")
        self.audio_list = glob(signal_path)
        self.info_type = (
            {
                "signal": "signal",
                "loudness": "loudness",
                "frequency": "frequency_c",
                "mfcc": "mfcc",
            }
            if frequency_with_confidence
            else 
            {
                "signal": "signal",
                "loudness": "loudness",
                "frequency": "frequency",
                "mfcc": "mfcc",
            }
        )

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        signal_path = self.audio_list[idx]
        file_name = signal_path.split("/")[-1][:-4]
        signal = np.load(
            os.path.join(self.dir_path, f"{self.info_type['signal']}/{file_name}.npy")
        ).astype("float32")
        loudness = np.load(
            os.path.join(self.dir_path, f"{self.info_type['loudness']}/{file_name}.npy")
        ).astype("float32")[..., :-1]
        frequency = np.load(
            os.path.join(self.dir_path, f"{self.info_type['frequency']}/{file_name}.npy")
        ).astype("float32")
        # mfcc = np.load(
        #     os.path.join(self.dir_path, f"{self.info_type['mfcc']}/{file_name}.npy")
        # ).astype("float32")

        return (file_name, signal, loudness, frequency)