# TimbreTransformer

基於 DDSP 架構的音色轉換模型，透過可微分數位訊號處理（Differentiable Digital Signal Processing）實現樂器音色遷移。

## 模型架構

TimbreTransformer 採用 Encoder-Decoder 架構搭配 DDSP 合成器：

### Encoder
- **Source Encoder**：從音訊訊號中提取基頻（F0）、響度（Loudness）、能量（Energy）
  - F0 提取使用預訓練 CREPE 模型
  - 響度提取使用 A-weighting 功率譜
  - Energy 使用 frame-level RMS
- **Timbre Encoder**（`TimbreEncoderX`）：從目標音色訊號提取 timbre embedding
  - 基於 Mel Spectrogram + ResBlock 的 VAE 架構
  - 輸出 μ 和 logvar，使用 reparameterization trick 採樣

### Decoder（decoder_v21）
- **F0 / Loudness MLP**：分別將 F0 和 loudness 映射至 embedding 空間
- **Timbre Z Generator**：融合 timbre embedding 與 energy 特徵
  - 內含 TimbreFusionBlock（gate mechanism）和 TimbreAffineBlock（affine transformation）
- **GRU Mixer**：混合 F0、loudness、timbre 的 embedding
- **Timbre Transformer**：使用 self-attention + cross-attention 進一步融合特徵
- **輸出頭**：
  - **Harmonic Head**：101 個諧波分布 + 全域振幅
  - **Noise Head**：65 個 noise filter bank
  - **Enhance Harmonic Head**（NonIntHarmonicHead）：40 維非整數諧波係數

### Synthesizer
- **Harmonic Oscillator**：基於諧波加法合成（additive synthesis）
- **Noise Filter**：基於 FFT 卷積的噪音濾波器（subtractive synthesis）
- **Enhance Harmonic Oscillator**：非整數倍頻諧波合成（增強高頻細節）
- 最終輸出 = additive + subtractive + enhance harmonic

### Discriminator
- **Multi-Period Discriminator（MPD）**：來自 BigVGAN，使用 periods [2, 3, 5, 7, 11] 進行對抗訓練

## 專案結構

```
timber_transfer/
├── components/
│   ├── timbre_transformer/
│   │   ├── TimberTransformer.py    # 主模型（TimbreTransformer）
│   │   ├── TimbreFusionAE.py       # Timbre Fusion AutoEncoder（推論用）
│   │   ├── encoder.py              # Timbre Encoder、Source Encoder
│   │   ├── component.py            # DDSP 合成器元件
│   │   ├── utils.py                # 音訊特徵提取工具
│   │   ├── decoders/               # Decoder 版本（v0 ~ v21）
│   │   │   ├── __init__.py         # 目前使用 decoder_v21
│   │   │   └── decoder_v21.py      # 最終版本 Decoder
│   │   ├── utils_blocks/           # 網路區塊元件
│   │   │   ├── TCUB.py             # Attention 相關區塊
│   │   │   └── UFB.py              # Affine / DFBlock / UpFusionBlock
│   │   └── ptcrepe/                # 預訓練 CREPE 模型
│   ├── discriminators.py           # MPD / MRD 判別器
│   └── utils.py
├── data/
│   ├── dataset.py                  # NSynth Dataset
│   └── data_processor.py           # 資料前處理（提取 F0、loudness、MFCC）
├── app/
│   ├── app.py                      # Gradio Demo（重建展示）
│   └── timbre_transfer_app.py      # Gradio Demo（音色轉換展示）
├── tools/
│   ├── utils.py                    # 訓練工具函式
│   ├── loss_collector.py           # Loss 函式集合
│   ├── visual.py                   # 視覺化工具
│   └── cal_model_size.py           # 模型參數量計算
├── preceptual_loss/                # 基於 CREPE 的感知損失
├── experimental/                   # 實驗性視覺化腳本
├── ddsp_ori/                       # 原始 DDSP 參考實作
├── pt_file/                        # 模型權重檔案
├── record/                         # 驗證記錄
├── train.py                        # 訓練腳本（最終版本）
├── train_exp.py                    # 訓練腳本（實驗版本）
├── validation.py                   # 驗證腳本（loudness L1 + pitch L1）
├── valid_record.py                 # 驗證記錄腳本
├── config.json                     # 超參數設定
└── validation_log.txt              # 驗證結果記錄
```

## 資料集

使用 [NSynth](https://magenta.tensorflow.org/datasets/nsynth) 子集，預處理後的資料結構：

```
nsynth-subset/
├── train/
│   ├── signal/          # 音訊波形（.npy, 16kHz, 4 秒）
│   ├── frequency/       # F0 序列（.npy）
│   ├── frequency_c/     # F0 + confidence（.npy）
│   └── loudness/        # 響度序列（.npy）
├── valid/
└── test/
```

### 資料前處理

```bash
# 在 data/ 目錄下執行
python data_processor.py
```

`DataProcessor` 可生成：
- `frequency_c`：使用 CREPE 提取 F0 + confidence
- `loudness`：使用 A-weighting 提取響度

## 訓練

### 超參數（config.json）

| 參數 | 值 |
|------|-----|
| Learning Rate | 0.0002 |
| Optimizer | AdamW (β1=0.8, β2=0.99) |
| LR Scheduler | ExponentialLR (γ=0.999) |
| Batch Size | 16 |
| Epochs | 300 |
| Sampling Rate | 16000 Hz |
| n_fft | 1024 |
| hop_size | 256 |

### Loss 函式

| Loss | 權重 |
|------|------|
| Multi-scale FFT Loss | 3 |
| Mel Spectrogram L1 Loss | 45 |
| KL Divergence Loss | 0.01 |
| Feature Matching Loss (MPD) | 2 |
| GAN Generator Loss | 1 |

### 開始訓練

```bash
python train.py
```

訓練過程使用 [Weights & Biases](https://wandb.ai/) 記錄 loss 曲線，專案名稱為 `TimbreTransformer_v4`。

## 驗證

驗證使用兩個指標：
- **Loudness L1 Loss**：重建訊號與原始訊號的響度差異
- **Pitch L1 Loss**：重建訊號與原始訊號的基頻差異（以 MIDI 為單位，confidence threshold=0.85）

```bash
python validation.py
```

結果會記錄在 `validation_log.txt`。

## Demo 應用

### 重建展示

```bash
cd app && python app.py
```

隨機從資料集取樣一筆音訊，通過模型重建後比較原始與重建訊號。

### 音色轉換展示

```bash
cd app && python timbre_transfer_app.py
```

可分別選擇 source 和 target 音訊，自由搭配 timbre / loudness / F0 的來源，進行音色轉換。

## 環境需求

- Python 3.8+
- PyTorch
- torchaudio
- librosa
- torchcrepe
- gradio
- wandb
- numpy
- tqdm
- matplotlib
