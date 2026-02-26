# Source Encoder（來源編碼器）

[English](./source-encoder.md) | **繁體中文**

## 架構圖

![Source Encoder](../../assets/architecture/source_encoder.png)

## 設計動機

Source Encoder 的目標是在音色轉換過程中保留「演奏內容」：

- `F0`：維持旋律與音高軌跡
- Loudness：維持強弱起伏與包絡
- RMS Energy：提供 frame-level 能量條件，輔助解碼端融合

這對應論文的核心要求：改變音色，不破壞原演奏表情。

## 資料流

1. 由來源波形抽取控制特徵（可離線或在線）
2. 將 frame 級控制量整理成模型輸入 tensor
3. 將 `F0`、loudness、energy 傳給 decoder

## 程式碼對照

- Pitch 抽取（CREPE）
  - `components/timbre_transformer/utils.py`
  - `get_extract_pitch_needs`, `extract_pitch`
- Loudness 抽取（A-weighted STFT power）
  - `components/timbre_transformer/utils.py`
  - `get_A_weight`, `extract_loudness`
- 執行期 source encoder 包裝
  - `components/timbre_transformer/encoder.py`
  - `Encoder.forward`, `EngryEncoder.forward`

## Tensor 介面

- Input
  - `signal`: `(B, T)`
  - `loudness`: `(B, F)`
  - `f0`: `(B, F)`
- Output
  - `f0`: `(B, F, 1)`
  - `loudness`: `(B, F, 1)`
  - `energy`: `(B, F, 1)`

## 與指標/訓練的關聯

$$
\mathcal{L}_{\text{loudness}} = \left\lVert l(x)-l(\hat{x}) \right\rVert_1
$$

$$
\mathcal{L}_{f0} = \operatorname{mean}\left(\left|midi(f0(x)) - midi(f0(\hat{x}))\right|\right)
$$

目前程式：
- `train.py`：loudness L1 為監控項
- `validation.py`：loudness L1 / pitch L1 為評估項
