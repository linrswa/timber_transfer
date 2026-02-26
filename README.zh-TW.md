# TimbreTransformer — 以 GAN 為基礎的樂器音色轉換

[English](./README.md) | **繁體中文**

<p align="center">
  <img src="assets/architecture/timbre_transfer_concept.png" width="600"/>
</p>

**TimbreTransformer** 是一個基於 GAN 與 DDSP（可微分數位訊號處理）的任意對任意樂器音色轉換系統。給定一段**來源音訊（source）**與一段不同樂器的**參考音訊（reference）**，模型會在保留音高、響度與演奏表情的前提下，將來源音訊轉換為參考音色。

> 論文：*Generative Adversarial Network Design for Instrument Timbre Transfer Application*  
> （應用於樂器音色轉換之生成對抗網路設計）

---

## 特色

- **任意對任意音色轉換**：不需針對每組樂器重新訓練
- **DDSP 合成架構**：以整數諧波 + 非整數諧波 + 濾波噪聲組成可解釋的聲音生成
- **輕量化模型**：約 **7.5M** 參數，接近 DDSP（7M），遠小於 WaveRNN（23M）
- **高品質重建**：Loudness L1 ≈ 0.08、F0 L1 ≈ 0.02（MIDI 尺度）

---

## 模型架構

<p align="center">
  <img src="assets/architecture/model_architecture.png" width="700"/>
</p>

模型由四個主要元件組成：

| 元件 | 角色 |
|---|---|
| **Source Encoder** | 從來源音訊抽取 F0（透過預訓練 CREPE）、A-weighted loudness、RMS energy |
| **Timbre Encoder** | 以 VAE 方式由參考音訊 mel spectrogram 抽取 timbre embedding（Timbre Z） |
| **Decoder** | 透過 GRU + Timbre Transformer（self/cross attention）融合來源特徵與 timbre embedding，輸出三個合成 head 參數 |
| **DDSP Synthesizer** | 使用整數諧波 + 非整數諧波 + 濾波噪聲三分支生成最終音訊 |

對抗式訓練採用來自 HiFi-GAN 的 **MPD（Multi-Period Discriminator）**。

### 元件文件（中文）

- [元件總覽](docs/components/README.zh-TW.md)
- [Source Encoder（來源編碼器）](docs/components/source-encoder.zh-TW.md)
- [Timbre Encoder（音色編碼器）](docs/components/timbre-encoder.zh-TW.md)
- [Decoder（解碼器）](docs/components/decoder.zh-TW.md)
- [DDSP Synthesizer（DDSP 合成器）](docs/components/ddsp-synthesizer.zh-TW.md)

每份元件文件皆包含：
- 架構圖
- 設計動機（對齊論文）
- 資料流與 tensor 介面
- 程式碼對照
- 數學公式
- 訓練目標與 loss 關聯

<details>
<summary><b>Decoder 架構圖（展開）</b></summary>
<p align="center">
  <img src="assets/architecture/decoder.png" width="650"/>
</p>
</details>

---

## 評估結果

### 重建品質（與 baseline 比較）

| 模型 | Loudness L1 ↓ | F0 L1 ↓ | 參數量 |
|---|---|---|---|
| WaveRNN | 0.10 | 1.00 | 23M |
| DDSP | 0.07 | 0.02 | 7M |
| **Ours** | **0.08** | **0.02** | **7.5M** |

本模型在重建品質上接近 DDSP，同時具備 DDSP 不具備的 **any-to-any 音色轉換**能力。

### 重建可視化

<p align="center">
  <img src="assets/results/resynthesis_flute.png" width="45%"/>
  <img src="assets/results/resynthesis_brass.png" width="45%"/>
</p>
<p align="center">
  <img src="assets/results/resynthesis_mallet.png" width="45%"/>
  <img src="assets/results/resynthesis_reed.png" width="45%"/>
</p>

每張圖包含：波形（ori vs rec）、loudness/F0 差異、mel spectrogram 對照。

---

## 資料集

使用 [NSynth](https://magenta.tensorflow.org/datasets/nsynth) 子集（70,379 筆）訓練，涵蓋 strings、brass、woodwinds、percussion。所有音檔為 16 kHz、4 秒單音，MIDI pitch 範圍 24–84。

---

## 快速開始

### 環境需求

```bash
Python 3.8+, PyTorch, torchaudio, librosa, torchcrepe, gradio, wandb
```

### 前處理

```bash
cd data && python data_processor.py
```

### 訓練

```bash
python train.py
```

訓練紀錄會寫入 [Weights & Biases](https://wandb.ai/) 的 `TimbreTransformer_v4` 專案。

### Gradio Demo

```bash
# 重建 demo
cd app && python app.py

# 音色轉換 demo
cd app && python timbre_transfer_app.py
```

---

## 專案結構

```text
timber_transfer/
├── components/
│   └── timbre_transformer/      # 核心模型（encoder, decoder, synthesizer）
├── data/                        # NSynth 與前處理
├── app/                         # Gradio demo
├── tools/                       # 訓練工具、loss、可視化
├── train.py                     # 訓練腳本
├── validation.py                # 評估（Loudness L1, Pitch L1）
└── assets/                      # 架構圖與結果圖
```

---

## License

本專案為碩士論文研究實作。

## Acknowledgements

本專案參考 [DDSP](https://github.com/magenta/ddsp)、[HiFi-GAN](https://github.com/jik876/hifi-gan)、[NVC-Net](https://github.com/sony/ai-research-code) 的設計概念。
