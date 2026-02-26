# DDSP Synthesizer

**English** | [繁體中文](./ddsp-synthesizer.zh-TW.md)

## Position in Full Model

![Model Architecture](../../assets/architecture/model_architecture.png)

The synthesizer is the final signal-generation stage after decoder heads.

## Design Intent

Use interpretable differentiable DSP branches instead of black-box waveform generation.

- Integer harmonic oscillator branch: periodic tonal core.
- Filtered noise branch: aperiodic/noisy components.
- Non-integer harmonic branch: additional inharmonic detail for richer timbre.

Final reconstruction:

`reconstruct_signal = additive_output + subtractive_output + enhance_harmonic_output`

## Branch Details

1. Integer Harmonic Branch
- Module: `HarmonicOscillator`
- Input: `(n_harm_dis, global_amp)`, `f0`
- Key operations:
  - remove harmonics above Nyquist
  - frame-to-sample upsampling
  - sinusoidal additive synthesis

2. Filtered Noise Branch
- Module: `NoiseFilter`
- Input: noise filter-bank magnitudes
- Key operations:
  - magnitude-to-impulse-response conversion
  - FFT convolution with white noise

3. Non-Integer Harmonic Branch
- Module: `EnhanceHarmonicOscillator`
- Input: `(n_harm_dis, global_amp, enhance_coef)`, `f0`
- Key operations:
  - inharmonic coefficient conditioning
  - Nyquist masking
  - additive synthesis in enhanced harmonic space

## Implementation Mapping

- Signal generation modules:
  - `components/timbre_transformer/component.py`
  - `HarmonicOscillator`, `NoiseFilter`, `EnhanceHarmonicOscillator`
- Model integration:
  - `components/timbre_transformer/TimberTransformer.py`
  - `TimbreTransformer.forward`

## Why This Matters for Thesis Goals

This design gives strong controllability and interpretability: each branch has a clear acoustic role, making analysis and ablation easier than end-to-end waveform-only generators.

## Mathematical Formulation

### Integer Harmonic Branch

For frame/sample index \(t\), with \(K\) harmonics:

\[
x_{\text{harm}}(t) = \sum_{k=1}^{K} a_k(t)\sin(\phi_k(t)),
\quad
\phi_k(t)=\sum_{\tau \le t}2\pi k f_0(\tau)/f_s
\]

with Nyquist masking (\(k f_0 < f_s/2\)).

### Noise Branch

Predicted magnitude response \(n_t\) is converted to impulse response \(h_t\), then:

\[
x_{\text{noise}} = w * h
\]

where \(w\) is white noise and \(*\) is FFT convolution.

### Enhancement Branch

The enhancement head predicts additional harmonic controls
\((\tilde{h}, \tilde{a}, \alpha)\), then synthesizer generates:

\[
x_{\text{enh}}(t) = \sum_{k=1}^{K} \tilde{a}_k(t)\sin(\tilde{\phi}_k(t))
\]

In current implementation, \(\alpha\) is used in masking/conditioning before synthesis.

### Final Reconstruction

\[
\hat{x}(t)=x_{\text{harm}}(t)+x_{\text{noise}}(t)+x_{\text{enh}}(t)
\]

## Loss Coupling in Training

Synthesizer receives no direct branch-level target. It is optimized end-to-end through:
- adversarial and feature matching losses on waveform realism
- mel and multi-scale FFT losses on spectral fidelity

This forces branch decomposition to be useful for reconstruction and transfer quality under the full objective.
