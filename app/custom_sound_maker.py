#%%
import numpy as np
from scipy.io import wavfile
import torch
import sys
sys.path.append('..')
from components.timbre_transformer.utils import get_A_weight, extract_loudness 
from components.timbre_transformer.utils import get_extract_pitch_needs, extract_pitch 

class CustomSoundMaker():
    def __init__(self):
        # Define the note frequencies
        self.note_frequencies = {
            'Do': 261.63,
            'Re': 293.66,
            'Mi': 329.63,
            'Fa': 349.23,
            'So': 392.00,
            'La': 440.00,
            'Ti': 493.88,
            'Do_octave': 523.25
        }

        self.duration = 1.0  # Duration of each note in seconds
        self.fade_out = 0.2  # Fade-out duration in seconds
        self.sample_rate = 16000  # Number of samples per second
        self.play_notes = ['So', 'Mi', 'Mi', 'Fa']

    def create_sound(self):
        # Generate the time axis for one note
        num_samples_per_note = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, num_samples_per_note, endpoint=False)

        # Initialize an empty waveform for the entire sequence
        total_duration = len(self.play_notes) * self.duration
        total_samples = int(total_duration * self.sample_rate)
        waveform = np.zeros(total_samples)

        for i, note in enumerate(self.play_notes):
            frequency = self.note_frequencies[note]
            
            note_waveform = np.sin(2 * np.pi * frequency * t)
            
            # Apply fade-out to the note waveform
            fade_out_samples = int(self.fade_out * self.sample_rate)
            fade_out_window = np.linspace(1, 0, fade_out_samples)
            note_waveform[-fade_out_samples:] *= fade_out_window
            
            # Calculate the start and end positions in the overall waveform
            start = i * num_samples_per_note
            end = start + num_samples_per_note
            
            # Add the note waveform to the overall waveform
            waveform[start:end] = note_waveform

        # Ensure the waveform is clipped to the total duration
        waveform = waveform[:total_samples]

        # # Scale the waveform to 16-bit integer range (-32768 to 32767)
        scaled_waveform = np.int16(waveform * 32767)
        print(scaled_waveform.shape)  # Should print (64000,)

        waveform_tensor = torch.tensor(waveform).float().unsqueeze(0)
        loudness = self.get_loudness(waveform_tensor)
        pitch = self.get_pitch(waveform_tensor)
        np.save('loudness.npy', loudness[:, :-1])
        np.save('pitch.npy', pitch[:, :-1])
        wavfile.write('sound.wav', self.sample_rate, scaled_waveform)
    
    def get_loudness(self, waveform):
        A_weight = get_A_weight()
        loudness = extract_loudness(waveform, A_weight)
        return loudness

    def get_pitch(self, waveform):
        device, cr_model, m_sec = get_extract_pitch_needs(device="cpu")
        pitch = extract_pitch(waveform, device=device, cr=cr_model, m_sec=m_sec)
        return pitch

sound_maker = CustomSoundMaker() 
sound_maker.create_sound()

import matplotlib.pyplot as plt

# Load the loudness and pitch data
_, sound = wavfile.read('sound.wav')
loudness = np.load('loudness.npy').squeeze()
pitch = np.load('pitch.npy').squeeze()

# Plot sound
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(sound)
plt.title('sonud')
plt.xlabel('Sample')
plt.ylabel('amplitude')

# Plot loudness
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(loudness)
plt.title('Loudness')
plt.xlabel('Sample')
plt.ylabel('Loudness')

# Plot pitch
plt.subplot(3, 1, 2)
plt.plot(pitch)
plt.title('Pitch')
plt.xlabel('Sample')
plt.ylabel('Pitch')

plt.tight_layout()
plt.show()