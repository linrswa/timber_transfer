import numpy as np
from scipy.io import wavfile

# Define the note frequencies
note_frequencies = {
    'Do': 261.63,
    'Re': 293.66,
    'Mi': 329.63,
    'Fa': 349.23
}

duration = 1.0  # Duration of each note in seconds
fade_out = 0.2  # Fade-out duration in seconds
sample_rate = 16000  # Number of samples per second

# Generate the time axis for one note
num_samples_per_note = int(duration * sample_rate)
t = np.linspace(0, duration, num_samples_per_note, endpoint=False)

# Initialize an empty waveform for the entire sequence
total_duration = len(note_frequencies) * duration
total_samples = int(total_duration * sample_rate)
waveform = np.zeros(total_samples)

# Generate the waveform for each note and add it to the overall waveform
for i, note in enumerate(['Do', 'Re', 'Mi', 'Fa']):
    frequency = note_frequencies[note]
    
    # Generate the sine wave for the note
    note_waveform = np.sin(2 * np.pi * frequency * t)
    
    # Apply fade-out to the note waveform
    fade_out_samples = int(fade_out * sample_rate)
    fade_out_window = np.linspace(1, 0, fade_out_samples)
    note_waveform[-fade_out_samples:] *= fade_out_window
    
    # Calculate the start and end positions in the overall waveform
    start = i * num_samples_per_note
    end = start + num_samples_per_note
    
    # Add the note waveform to the overall waveform
    waveform[start:end] = note_waveform

# Ensure the waveform is clipped to the total duration
waveform = waveform[:total_samples]

# Scale the waveform to 16-bit integer range (-32768 to 32767)
scaled_waveform = np.int16(waveform * 32767)
print(scaled_waveform.shape)  # Should print (64000,)

# Save the waveform as a WAV file
wavfile.write('sound.wav', sample_rate, scaled_waveform)
