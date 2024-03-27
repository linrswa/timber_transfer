#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from components.timbre_transformer.component import HarmonicOscillator

# Define t_torch
t_torch = torch.linspace(0, 1, steps=32000)
# Create three different frequency and amplitude sin waves using torch
wave1 = torch.sin(2 * np.pi * 5 * t_torch)
wave2 = 0.5 * torch.sin(2 * np.pi * 10 * t_torch)
wave3 = 0.25 * torch.sin(2 * np.pi * 15 * t_torch)

# Add them together
wave_combined = wave1 + wave2 + wave3

# Convert the combined wave back to numpy for plotting
wave_combined_np = wave_combined.numpy()

wave_after_smooth = HarmonicOscillator.smooth_envelop(wave_combined.view(1, -1, 1))
# Convert wave_after_smooth back to numpy for plotting
wave_after_smooth_np = wave_after_smooth.numpy().squeeze()

# Create subplots
fig, axs = plt.subplots(2)

# Plot the combined wave
axs[0].plot(wave_combined_np)
axs[0].set_title('Combined Wave')

# Plot the smoothed wave
axs[1].plot(wave_after_smooth_np)
axs[1].set_title('Smoothed Wave')

# Show the plots
plt.tight_layout()
plt.show()



# %%
