# Decoder（解碼器）

[English](./decoder.md) | **繁體中文**

## 架構圖

![Decoder](../../assets/architecture/decoder.png)

### Timbre Transformer 區塊

![Timbre Transformer Block](../../assets/architecture/components/timbre-transformer-block.png)

## 設計動機

Decoder 是整個模型的融合核心，負責把來源控制資訊與音色向量結合，輸出可合成的聲學參數。

主要設計：
- `F0` 與 loudness 控制流編碼
- 帶有 energy 條件的 Timbre-Z 調制
- GRU 進行時間建模
- self-attention + cross-attention 進行內容與音色融合

## 輸出 Head

### Harmonic Head（整數諧波）

![Harmonic Head](../../assets/architecture/components/harmonic-head.png)

### Noise Head（濾波噪聲）

![Noise Head](../../assets/architecture/components/noise-head.png)

### Non-Integer Harmonic Head（非整數諧波）

![Non-Integer Harmonic Head](../../assets/architecture/components/non-integer-head.png)

## 程式碼對照

- 入口
  - `components/timbre_transformer/decoders/__init__.py`
  - 目前導向 `decoder_v21.Decoder`
- 核心邏輯
  - `components/timbre_transformer/decoders/decoder_v21.py`
- 輔助區塊
  - `components/timbre_transformer/utils_blocks/TCUB.py`
  - `components/timbre_transformer/utils_blocks/UFB.py`

## Tensor 介面

- Input
  - `f0`: `(B, F, 1)`
  - `loudness`: `(B, F, 1)`
  - `energy`: `(B, F, 1)`
  - `timbre_emb`: `(B, 1, D)`
- Output
  - `harmonic_output`: `(n_harm_dis, global_amp)`
  - `noise_output`: noise filter-bank magnitudes
  - `enhance_harmonic_output`: `(n_harm_dis, global_amp, enhance_coef)`
  - `f0`（傳遞給 synthesizer）

## 參數約束

Head 端使用 `modified_sigmoid` 保證振幅與濾波器幅值非負，提升 DDSP 合成穩定性。

## 數學公式

令來源控制為 $c_t=[f0_t, l_t, e_t]$，音色條件為 $z$，則 decoder 預測：

$$
\theta_t = \{a_t, h_t, n_t, \tilde{h}_t, \tilde{a}_t, \alpha_t\}
$$

其中：
- $h_t$：整數諧波分佈
- $a_t$：全域諧波振幅
- $n_t$：噪聲濾波器幅值
- $(\tilde{h}_t,\tilde{a}_t,\alpha_t)$：增強分支參數

非負參數化：

$$
\operatorname{msig}(x)=m\cdot \sigma(x)^{\log(e)}+\tau
$$

（程式預設 $m=2, e=10, \tau=10^{-7}$）。

諧波分佈正規化：

$$
h_t \leftarrow \frac{h_t}{\sum_k h_{t,k} + \varepsilon}
$$

## 與 loss 的關聯

Decoder 沒有 head-level 直接標註，主要透過合成後波形的間接監督學習：
- 對抗損失 $\mathcal{L}_{adv}^G$
- Feature matching $\mathcal{L}_{fm}$
- Mel 重建 $\mathcal{L}_{mel}$
- Multi-scale FFT $\mathcal{L}_{mfft}$
