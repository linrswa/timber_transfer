# DDSP Synthesizer（DDSP 合成器）

[English](./ddsp-synthesizer.md) | **繁體中文**

## 在整體模型中的位置

![Model Architecture](../../assets/architecture/model_architecture.png)

Synthesizer 位於 decoder 之後，負責將參數轉為最終波形。

## 設計動機

以可解釋的可微分 DSP 分支取代黑箱式波形生成：

- 整數諧波分支：負責週期性主體音色
- 濾波噪聲分支：負責非週期/噪聲成分
- 非整數諧波分支：補強 inharmonic 細節

最終重建：

`reconstruct_signal = additive_output + subtractive_output + enhance_harmonic_output`

## 分支細節

1. 整數諧波分支
- Module: `HarmonicOscillator`
- Input: `(n_harm_dis, global_amp)`, `f0`
- 核心步驟：
  - Nyquist 以上諧波抑制
  - frame-to-sample 上採樣
  - 正弦加法合成

2. 濾波噪聲分支
- Module: `NoiseFilter`
- Input: noise filter-bank magnitudes
- 核心步驟：
  - 幅度譜轉 impulse response
  - 白噪聲與 impulse response 做 FFT convolution

3. 非整數諧波分支
- Module: `EnhanceHarmonicOscillator`
- Input: `(n_harm_dis, global_amp, enhance_coef)`, `f0`
- 核心步驟：
  - 非整數係數條件化
  - Nyquist masking
  - 增強諧波域加法合成

## 程式碼對照

- 合成模組
  - `components/timbre_transformer/component.py`
  - `HarmonicOscillator`, `NoiseFilter`, `EnhanceHarmonicOscillator`
- 整合位置
  - `components/timbre_transformer/TimberTransformer.py`
  - `TimbreTransformer.forward`

## 數學公式

### 整數諧波分支

$$
x_{\text{harm}}(t) = \sum_{k=1}^{K} a_k(t)\sin(\phi_k(t)),
\quad
\phi_k(t)=\sum_{\tau \le t}2\pi k f_0(\tau)/f_s
$$

並施加 Nyquist 條件 $k f_0 < f_s/2$。

### 噪聲分支

$$
x_{\text{noise}} = w * h
$$

其中 $w$ 為白噪聲、$*$ 表卷積（程式採 FFT convolution）。

### 非整數諧波分支

$$
x_{\text{enh}}(t) = \sum_{k=1}^{K} \tilde{a}_k(t)\sin(\tilde{\phi}_k(t))
$$

目前實作中，$\alpha$ 用於合成前的 masking/conditioning。

### 最終輸出

$$
\hat{x}(t)=x_{\text{harm}}(t)+x_{\text{noise}}(t)+x_{\text{enh}}(t)
$$

## 與 loss 的關聯

Synthesizer 沒有分支級別直接 supervision，透過 end-to-end loss 學習：
- 對抗 + feature matching（提升波形真實感）
- mel + multi-scale FFT（維持頻譜保真）
