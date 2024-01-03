#%% 
import os
import torch
import torchaudio
from torch.utils.data import Dataset ,DataLoader
import numpy as np
from glob import glob
from tqdm import tqdm

import sys
sys.path.append(".")
from components.timbre_transformer.utils import extract_pitch, get_extract_pitch_needs

class DatasetForProcessor(Dataset):
    def __init__(
        self,
        data_mode: str,
        dataset_dir: str = "/dataset/NSynth/nsynth-subset",
        sr: int = 16000,
    ):
        super().__init__()
        self.sr = sr
        self.dir_path = f"{dataset_dir}/{data_mode}"
        signal_path = os.path.join(self.dir_path, "signal/*")
        self.audio_list = glob(signal_path)

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        signal_path = self.audio_list[idx]
        file_name = signal_path.split("/")[-1][:-4]
        signal = np.load(os.path.join(self.dir_path, f"signal/{file_name}.npy")).astype(
            "float32"
        )

        return file_name, signal

class DataProcessor:

    def __init__(
        self,
        data_mode: str,
        dataset_dir: str = "/dataset/NSynth/nsynth-subset",
        sr: int = 16000,
        n_fft: int = 1024,
        source: str = "signal",
    ):
        self.data_mode = data_mode
        self.dir_path = f"{dataset_dir}/{data_mode}"
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = int(n_fft / 4)
        self.save_dir_dict = {
            "signal": "signal",
            "frequency_c": "frequency_c",
            "loudness": "loudness",
            "mfcc": "mfcc",
        }
        self.source = self._check_source(source)
        self.source_list = self._get_source_path_list()
        self._check_folders_isexist()

    def _check_source(self, source: str):
        if source not in ["audio", "signal"]:
            raise ValueError(
                f"source must be 'audio(.wav)' or 'signal(.npy)', but {source}"
            )
        return source

    def _check_folders_isexist(self):
        save_dir_list = list(self.save_dir_dict.values())
        for folder_name in save_dir_list:
            if not os.path.isdir(os.path.join(self.dir_path, folder_name)):
                os.mkdir(os.path.join(self.dir_path, folder_name))

    def _get_source_path_list(self):
        source_dir_path = os.path.join(self.dir_path, f"{self.source}/*")
        source_path_list = glob(source_dir_path)
        return source_path_list

    def _print_save_dir_info(self):
        print("Now save dir is:")
        for k, v in self.save_dir_dict.items():
            print(f"{self.dir_path}/{v}")

    def gen_mel_data(self):
        extract_mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=80,
            melkwargs=dict(
                n_fft=1024, hop_length=256, n_mels=128, f_min=20.0, f_max=8000.0
            )
        )
        
        self._print_save_dir_info()
        for source_path in tqdm(self.source_list):
            file_name = source_path.split("/")[-1][:-4]
            signal = np.load(source_path)
            signal = torch.from_numpy(signal)
            mfcc = extract_mfcc(signal)
            np.save(
                os.path.join(self.dir_path, f"{self.save_dir_dict['mfcc']}/{file_name}.npy"),
                mfcc.numpy(),
            )
    
    def extract_frequency_c(self):
        dl = DataLoader(DatasetForProcessor(self.data_mode), batch_size=32)
        self._print_save_dir_info()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device, cr, m_sec = get_extract_pitch_needs(device=device)
        for fn_batch, signal_batch in tqdm(dl):
            signal_batch = signal_batch.to(device)
            frequency_c_batch = extract_pitch(
                signal=signal_batch,
                device=device,
                cr=cr,
                m_sec=m_sec,
                sampling_rate=self.sr,
                with_confidence=True,
            )
            for fn, frequency_c in zip(fn_batch, frequency_c_batch):
                np.save(
                    os.path.join(self.dir_path, f"{self.save_dir_dict['frequency_c']}/{fn}.npy"),
                    frequency_c.numpy(),
                )


if __name__ == "__main__":
    processor = DataProcessor("valid")
    processor.extract_frequency_c()