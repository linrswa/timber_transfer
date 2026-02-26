# 元件文件總覽

[English](./README.md) | **繁體中文**

此區將主 README 的四大元件表格展開為架構層級文件。

## 全域架構

![Model Architecture](../../assets/architecture/model_architecture.png)

## 依元件閱讀

| 元件 | 設計目標 | 文件 |
|---|---|---|
| Source Encoder | 在音色轉換時穩定保留旋律與動態資訊 | [source-encoder.zh-TW.md](./source-encoder.zh-TW.md) |
| Timbre Encoder | 將參考音色壓縮成可控制的潛在向量 | [timbre-encoder.zh-TW.md](./timbre-encoder.zh-TW.md) |
| Decoder | 融合來源控制與音色嵌入並輸出合成參數 | [decoder.zh-TW.md](./decoder.zh-TW.md) |
| DDSP Synthesizer | 以可解釋的訊號分支把參數轉成波形 | [ddsp-synthesizer.zh-TW.md](./ddsp-synthesizer.zh-TW.md) |

## 對照依據

本文件對齊：
- 論文架構設計與章節論述
- `assets/architecture/` 與 `assets/architecture/components/` 圖表
- `components/timbre_transformer/` 目前程式實作

## 訓練目標總覽

目前 `train.py` 的最佳化目標為：

\[
\mathcal{L}_G =
\lambda_{\text{adv}}\mathcal{L}_{\text{adv}}^G +
\lambda_{\text{fm}}\mathcal{L}_{\text{fm}} +
\lambda_{\text{mel}}\mathcal{L}_{\text{mel}} +
\lambda_{\text{mfft}}\mathcal{L}_{\text{mfft}} +
\lambda_{\text{kl}}\mathcal{L}_{\text{kl}}
\]

\[
\mathcal{L}_D = \mathcal{L}_{\text{adv}}^D
\]

其中權重 \(\lambda_*\) 來自 `config.json -> loss_weight`。

### 對抗損失（MPD, LSGAN）

\[
\mathcal{L}_{\text{adv}}^D = \sum_p \mathbb{E}\left[(1-D_p(x))^2 + D_p(\hat{x})^2\right]
\]
\[
\mathcal{L}_{\text{adv}}^G = \sum_p \mathbb{E}\left[(1-D_p(\hat{x}))^2\right]
\]

### Feature Matching

\[
\mathcal{L}_{\text{fm}} = 2\sum_{p,l} \left\lVert f_{p,l}(x) - f_{p,l}(\hat{x}) \right\rVert_1
\]

### Mel 重建損失

\[
\mathcal{L}_{\text{mel}} = \left\lVert \text{Mel}(x) - \text{Mel}(\hat{x}) \right\rVert_1
\]

### Multi-Scale FFT 損失

\[
\mathcal{L}_{\text{mfft}} = \sum_{s \in \mathcal{S}}
\left(
\left\lVert S_s(x)-S_s(\hat{x}) \right\rVert_1
+
\left\lVert \log S_s(x)-\log S_s(\hat{x}) \right\rVert_1
\right)
\]

### KL 正則化（Timbre VAE）

\[
\mathcal{L}_{\text{kl}} = \frac{1}{2}\,\mathbb{E}
\left[
\sum_d \left(e^{\log\sigma_d^2} + \mu_d^2 -1-\log\sigma_d^2\right)
\right]
\]

### Loudness/F0 指標在程式中的位置

- `train.py`：loudness L1 以 `torch.no_grad()` 方式監控
- `validation.py`：loudness L1 與 pitch L1 為主要評估結果
