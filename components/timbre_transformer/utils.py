import torch
import numpy as np
import librosa as li
from tqdm import tqdm
from typing import Union
from .ptcrepe import crepe


def safe_log(x):
    return torch.log(x + 1e-7)

@torch.no_grad()
def mean_std_loudness(dataset):
    print("calculating mean and std of loudness...")
    mean = 0 
    std = 0
    n = 0
    for _, _, l, _ in tqdm(dataset):
        n += 1 
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    print(f"Doen! mean: {mean}, std: {std}")
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for scale in scales:
        S = torch.stft(
            input = signal,
            n_fft = scale,
            hop_length = int(scale * (1 - overlap)),
            win_length = scale,
            window = torch.hann_window(scale).to(signal),
            center = True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)
    
    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)
    return y

def get_A_weight(
    sampling_rate: int = 16000,
    n_fft: int = 1024,
    output_torch: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
    """Get a_weight from librosa.

    Args:
        sampling_rate (int, optional): sampling rate. Defaults to 16000.
        n_fft (int, optional): number of fft. Defaults to 1024.
        output_torch (bool, optional): output Tensor or not, if set this False will return ndarray. Defaults to True.

    Returns:
        ndarray or Tensor: Depends on output_torch.
    """
    f = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = li.A_weighting(f + 1e-20) 
    
    if output_torch:
        return torch.from_numpy(a_weight.reshape(-1, 1))
    
    return a_weight.reshape(-1, 1) # type(np.ndarray)

@torch.no_grad()
def extract_loudness(
    signal: torch.Tensor,
    a_weight: torch.Tensor,
    hop_length: int = 256,
    n_fft: int = 1024,
    ) -> torch.Tensor:
    """From a Tensor signal to a Tensor after loudness extraction.

    Args:
        signal (torch.Tensor): input shape should be (batch, frame)
        a_weight (torch.Tensor): input a_weight from get_a_weight() 
        hop_length (int, optional): n_fft/4 . Defaults to 256.
        n_fft (int, optional): number of fft. Defaults to 1024.

    Returns:
        torch.Tensor: return shape (batch, (frame/hop_length) + 1).
    """
    amplitude = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
        return_complex=True,
    )

    amin = 1e-20 # avoid divide by zero
    # amplitude to db
    power_db = 20.0 * torch.log10(torch.clamp(abs(amplitude), min=amin))
    # why '+' here? A: beacause it's log.
    loudness = power_db + a_weight
    
    # fix from https://github.com/acids-ircam/ddsp_pytorch/pull/32/commits/6fbe4d6eaabcbdfa5ad3597223323cb2962b1a58
    loudness = torch.mean(torch.pow(10, loudness / 10.0), dim=1)
    loudness = 10.0 * torch.log10(torch.clamp(loudness, min=amin))
    
    return loudness 

    
def get_extract_pitch_needs(
    device: str = "cuda",
    crepe_model: str = "full",
    hop_length: int = 256,
    sampling_rate: int = 16000,
    ) -> tuple:
    """ Get the needs for extract_pitch().

    Args:
        device (str, optional): Defaults to "cuda".
        crepe_model (str, optional): Defaults to "full".
        hop_length (int, optional): Defaults to 256.
        sampling_rate (int, optional): Defaults to 16000.

    Returns:
        tuple: (device, cr_model, m_sec) 
    """
    cr_model = crepe.CREPE(crepe_model).to(device)
    m_sec = int(hop_length * 1000/sampling_rate)
    return device, cr_model, m_sec

@torch.no_grad()
def extract_pitch(
    signal: torch.Tensor,
    device: str,
    cr: crepe.CREPE,
    m_sec: int,
    sampling_rate: int=16000,
    with_confidence: bool=False,
    ) -> torch.Tensor:
    """From a Tensor signal to a Tensor after pitch extraction.

    Args:
        signal (torch.Tensor): input shape should be (batch, frame)
        device (str): get from get_extract_pitch_needs().
        cr (crepe.CREPE): get from get_extract_pitch_needs().
        m_sec (int): get from get_extract_pitch_needs().
        sampling_rate (int, optional): Defaults to 16000.
        with_confidence (bool, optional): output confidence or not, if this set True return tuple(torch.Tensor, torch.Tensor). Defaults to False.

    Returns:
        torch.Tensor or tuple(torch.Tensor, torch.Tensor): Depends on with_confidence.
    """
    output = cr.predict(
        audio=signal.to(device),
        sr=sampling_rate,
        viterbi=False,
        center=True,
        step_size=m_sec,
        batch_size=256,
        )
    time, frequency, confidence, activation = output
        
    if with_confidence:
        frequency = frequency.unsqueeze(-1)
        confidence = confidence.unsqueeze(-1)
        return  torch.cat([frequency, confidence], dim=-1)
    else:
        return frequency