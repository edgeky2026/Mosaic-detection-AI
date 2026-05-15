# Mosaic-detection

モザイク漏れ検知 AI — 顔＋性器パイプライン

## 概要

動画コンテンツにおけるモザイク処理の品質を自動検査するシステム。
顔と性器の2つの独立パイプラインで検出→モザイク解析→判定を行い、統合判定を出力する。

## アーキテクチャ

```
動画入力
  ├── フレーム抽出 (ffmpeg, 1fps/0.5fps)
  ├── 顔パイプライン
  │   ├── SCRFD-2.5g/34g ハイブリッド検出 (施策D)
  │   │   └── 34g=1件時に 2.5g で複数顔を救済
  │   ├── ByteTrack トラッキング
  │   ├── 3指標スコアリング (Laplacian/ACF/DCT)
  │   └── Confidence Boost (施策B)
  ├── 性器パイプライン
  │   ├── Fine-tuned GroundingDino v3FT 検出
  │   │   └── text prompt: "vagina . penis . anus ."
  │   ├── 施策J: アスペクト比フィルタ (MAX 4.0)
  │   ├── 3指標スコアリング (低閾値)
  │   └── Area Filter (MIN 0.1%, MAX 30%)
  └── 統合判定 (FAIL > REVIEW > PASS)
```

## ディレクトリ構造

```
Mosaic-detection/
├── src/                    # メインソースコード
│   ├── config.py           # 全パラメータ集約
│   ├── pipeline.py         # 顔検知パイプライン
│   ├── genital_pipeline.py # 性器検知パイプライン
│   ├── combined_pipeline.py# 統合パイプライン (エントリポイント)
│   ├── scorer.py           # スコアリング・判定ロジック
│   ├── mosaic_analyzer.py  # ROI解析 (Laplacian/ACF/DCT)
│   ├── dino_checkpoint_utils.py  # GroundingDino checkpoint 自動解決
│   ├── eval_testset.py     # 顔検知精度評価
│   └── eval_genital.py     # 性器検知精度評価
├── vendor/                 # サードパーティライブラリ
│   ├── scrfd/              # SCRFD 顔検出 (ONNX)
│   ├── yolox/tracker/      # ByteTrack
│   └── grounding_dino/     # GroundingDino
├── models/                 # モデル重み (.gitignore対象)
│   ├── scrfd/
│   │   ├── scrfd_2.5g.onnx    # SCRFD-2.5GF (3.3MB)
│   │   └── scrfd_34g.onnx     # SCRFD-34GF (39MB)
│   └── gdino/
│       ├── v3ft_best.pth       # v3FT Epoch 7 best (1.2GB)
│       └── cfg_odvg.py
└── requirements.txt
```

## 使い方

### 単体動画の検査

```bash
python src/combined_pipeline.py --input video.mp4 --output result.json
```

### 顔のみ / 性器のみ

```bash
python src/combined_pipeline.py --input video.mp4 --face-only
python src/combined_pipeline.py --input video.mp4 --genital-only
```

### 精度評価

```bash
# 顔パイプライン (testset_2603)
TESTSET_ROOT=/path/to/testset_2603 python src/eval_testset.py --config all

# 性器パイプライン
TESTSET_ROOT=/path/to/testset_2603 python src/eval_genital.py
```

## モデル準備

モデル重みは `.gitignore` 対象のため別途配置が必要：

```bash
# SCRFD (顔検出)
models/scrfd/scrfd_2.5g.onnx   # InsightFace SCRFD-2.5GF
models/scrfd/scrfd_34g.onnx    # InsightFace SCRFD-34GF

# GroundingDino (性器検出) — v3FT best (Epoch 7)
models/gdino/v3ft_best.pth     # v3FT (264K images, lr=1e-6, mAP=0.4927)
models/gdino/cfg_odvg.py       # ODVG設定ファイル
```

## 精度指標 (2026-05-15)

### 顔パイプライン (testset_2603, n=2837)

| Config | Coverage | Precision | IoU |
|--------|----------|-----------|-----|
| 施策D hybrid (conf=0.3) | 0.6491 | 0.8024 | 0.5792 |
| Production (conf=0.5) | 0.5482 | 0.8463 | 0.5130 |

### 性器パイプライン (testset_2603, n=2353)

**v3FT Epoch 7 best** — box_threshold=0.18, text_threshold=0.15

| サブセット | n_eval | Coverage | Precision | IoU |
|:---:|:---:|:---:|:---:|:---:|
| normal | 835 | 0.711 | **0.884** | 0.633 |
| gay | 238 | 0.628 | **0.672** | 0.520 |
| hukusu | 662 | 0.401 | 0.409 | 0.281 |
| rez | 618 | 0.493 | **0.595** | 0.439 |
| **全体** | **2353** | **0.558** | **0.653** | **0.472** |

> 施策H（P=0.635）比で **Precision +0.018** の改善、目標 P ≥ 0.65 を達成。

### 精度推移

```
GCP baseline    : P=0.312  ████░░░░░░░░░░░░░░░░░░  (基準)
施策E (ローカル)  : P=0.480  ██████████████░░░░░░░░  (+54%)
施策H epoch 0   : P=0.635  ██████████████████░░░░  (+32%)
v3FT (epoch 7)  : P=0.653  ██████████████████░░░░  (+2.8%) ★BEST
顔 BiSeNet v6FT : P=0.824  ████████████████████████  参考
```

## 環境変数

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `TESTSET_ROOT` | テストデータセットルート | (ローカルパス) |
| `SEGFORMER_PROJ` | SegFormerプロジェクト | (ローカルパス) |

## Requirements

- Python 3.11+
- CUDA (NVIDIA GPU)
- ffmpeg
- onnxruntime-gpu
- torch >= 2.0
- opencv-python
- numpy
- scipy
- transformers
