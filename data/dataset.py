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
            ("signal", "loudness", "frequency_c")
            if frequency_with_confidence
            else ("signal", "loudness", "frequency")
        )

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        signal_path = self.audio_list[idx]
        file_name = signal_path.split("/")[-1][:-4]
        signal = np.load(
            os.path.join(self.dir_path, f"{self.info_type[0]}/{file_name}.npy")
        ).astype("float32")
        loudness = np.load(
            os.path.join(self.dir_path, f"{self.info_type[1]}/{file_name}.npy")
        ).astype("float32")[..., :-1]
        frequency = np.load(
            os.path.join(self.dir_path, f"{self.info_type[2]}/{file_name}.npy")
        ).astype("float32")

        return (file_name, signal, loudness, frequency)