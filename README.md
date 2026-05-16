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
├── models/                 # モデル重み (含む; *.pth は Git LFS 管理)
│   ├── scrfd/
│   │   ├── scrfd_2.5g.onnx    # SCRFD-2.5GF (3.2MB)
│   │   └── scrfd_34g.onnx     # SCRFD-34GF (38MB)
│   └── gdino/
│       ├── v3ft_best.pth       # v3FT Epoch 7 best (1.2GB, LFS)
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

モデル重みはリポジトリに含まれています（`*.pth` は Git LFS 管理）。

```bash
# SCRFD (顔検出)
models/scrfd/scrfd_2.5g.onnx   # InsightFace SCRFD-2.5GF
models/scrfd/scrfd_34g.onnx    # InsightFace SCRFD-34GF

# GroundingDino (性器検出) — v3FT best (Epoch 7)
models/gdino/v3ft_best.pth     # v3FT (264K images, lr=1e-6, mAP=0.4927) → Git LFS
models/gdino/cfg_odvg.py       # ODVG設定ファイル
```

> 大容量ファイル（`*.pth`）は Git LFS で管理されています。初回 clone 時は `git lfs pull` を実行してください。

## 精度指標 (2026-05-15)

### 顔パイプライン (testset_2603, n=2837)

| Config | Coverage (Recall) | Precision | F1 | IoU |
|--------|:-----------------:|:---------:|:--:|:---:|
| 施策D hybrid (conf=0.3) | 0.6491 | 0.8024 | **0.718** | 0.5792 |
| Production (conf=0.5) | 0.5482 | 0.8463 | **0.665** | 0.5130 |

### 性器パイプライン (testset_2603, n=2353)

**v3FT Epoch 7 best** — box_threshold=0.18, text_threshold=0.15

| サブセット | n_eval | Coverage (Recall) | Precision | F1 | IoU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| normal | 835 | 0.711 | **0.884** | **0.788** | 0.633 |
| gay | 238 | 0.628 | **0.672** | **0.649** | 0.520 |
| hukusu | 662 | 0.401 | 0.409 | 0.405 | 0.281 |
| rez | 618 | 0.493 | **0.595** | **0.539** | 0.439 |
| **全体** | **2353** | **0.558** | **0.653** | **0.602** | **0.472** |

> 施策H（P=0.635）比で **Precision +0.018** の改善、目標 P ≥ 0.65 を達成。

### FT 前後精度比較（性器パイプライン, testset_2603）

#### 全体推移（Precision・F1 重視）

| バージョン | 学習データ | **Precision** | **F1** | Coverage (Recall) | IoU | Precision 差分 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GCP baseline | 15K | 0.312 | 0.250 | 0.209 | 0.165 | 基準 |
| 施策E（ローカルFT） | 15K | 0.480 | 0.482 | 0.484 | 0.326 | +0.168 (+54%) |
| 施策H（大規模FT） | 264K | 0.635 | 0.616 | 0.598 | 0.478 | +0.155 (+32%) |
| **v3FT（Epoch 7）★** | **246K** | **0.653** | **0.602** | **0.558** | **0.472** | **+0.018 (+2.8%)** |

> GCP baseline から v3FT まで Precision は **+0.341（+109%）** の改善。

#### 施策H → v3FT サブセット別比較

| サブセット | 施策H Precision | v3FT Precision | **Δ Precision** | v3FT F1 | v3FT Coverage |
|:---:|:---:|:---:|:---:|:---:|:---:|
| normal | 0.876 | **0.884** | **+0.008** ✅ | 0.788 | 0.711 |
| gay | 0.662 | **0.672** | **+0.010** ✅ | 0.649 | 0.628 |
| hukusu | 0.424 | 0.409 | **−0.015** ⚠️ | 0.405 | 0.401 |
| rez | 0.578 | **0.595** | **+0.017** ✅ | 0.539 | 0.493 |
| **全体** | **0.635** | **0.653** | **+0.018** ✅ | **0.602** | **0.558** |

> hukusu（複数人物シーン）のみ若干低下。v4FT で targeted augmentation 対応予定。

#### Precision 推移（ASCII）

```
GCP baseline    : P=0.312  ████░░░░░░░░░░░░░░░░░░  (基準)
施策E (ローカル)  : P=0.480  ██████████████░░░░░░░░  (+54%)
施策H           : P=0.635  ██████████████████░░░░  (+32%)
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
