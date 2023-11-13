import torch
import json
from librosa.filters import mel as librosa_mel_fn

from data.dataset import NSynthDataset
from components.ddsp_modify.utils import mean_std_loudness

def get_mean_std_dict(data_mode: str, batch: int=32):
    mean_std_dict = {}
    dataset = NSynthDataset(data_mode=data_mode, sr=16000)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=8)
    mean_std_dict["mean_loudness"], mean_std_dict["std_loudness"]= mean_std_loudness(valid_loader)
    return mean_std_dict

def cal_loudness(loudness, mean_std_dict):
    return (loudness - mean_std_dict["mean_loudness"]) / mean_std_dict["std_loudness"]

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.squeeze(-1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_hyparam():
    """Get hyperparameters from config.json"""
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    with open("./config.json") as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)

    return h
    
