# Timbre Encoder

**English** | [繁體中文](./timbre-encoder.zh-TW.md)

## Architecture

![Timbre Encoder](../../assets/architecture/timbre_encoder.png)

### Latent Sampling (VAE)

![Timbre Encoder Sampling](../../assets/architecture/components/timbre-encoder-sampling.png)

## Design Intent

The timbre encoder extracts a compact latent code (`Timbre Z`) that captures instrument color from reference audio while being separable from source melody.

- `mu`, `logvar` model a latent distribution.
- Training uses reparameterization for stochastic sampling.
- Inference uses deterministic `mu` for stable conversion.

This matches the thesis design choice: use VAE-style latent representation for timbre controllability and robustness.

## Implementation Mapping

- Active timbre encoder path:
  - `components/timbre_transformer/encoder.py`
  - `TimbreEncoderX`
- Alternative encoder variant:
  - `TimbreEncoder`
- Sampling logic:
  - `components/timbre_transformer/TimberTransformer.py`
  - `TimbreTransformer.sample`

## Tensor Interfaces

- Input:
  - `timbre_signal`: reference waveform `(B, 1, T)` or `(B, T)` (pipeline-dependent)
- Encoder output:
  - `mu`: `(B, D, 1)`
  - `logvar`: `(B, D, 1)`
- Decoder conditioning form:
  - `timbre_emb`: `(B, 1, D)`

## Training vs Inference

- Training (`is_train=True`):
  - `z = mu + exp(0.5 * logvar) * eps`
- Inference (`is_train=False`):
  - `z = mu`

This keeps training expressive but inference consistent.

## Mathematical Formulation

Given reference audio $x_r$, the timbre encoder predicts:

$$
q_\phi(z \mid x_r) = \mathcal{N}(\mu_\phi(x_r), \text{diag}(\sigma_\phi^2(x_r)))
$$

Reparameterization:

$$
\epsilon \sim \mathcal{N}(0, I), \quad
z = \mu + \sigma \odot \epsilon, \quad
\sigma = \exp(0.5\,\logvar)
$$

Inference mode uses $z=\mu$.

## Loss Coupling in Training

The timbre encoder is mainly regularized by KL loss:

$$
\mathcal{L}_{\text{kl}} = \frac{1}{2}\,\mathbb{E}
\left[
\sum_d \left(e^{\log\sigma_d^2} + \mu_d^2 -1-\log\sigma_d^2\right)
\right]
$$

This term constrains latent space smoothness and improves timbre interpolation/generalization.
