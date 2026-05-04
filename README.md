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
  │   ├── ByteTrack トラッキング
  │   ├── 3指標スコアリング (Laplacian/ACF/DCT)
  │   └── Confidence Boost (施策B)
  ├── 性器パイプライン
  │   ├── Fine-tuned GroundingDino 検出
  │   ├── 3指標スコアリング (低閾値)
  │   └── Area Filter (MAX 30%)
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
│       ├── dino_local_ft_ep4_best.pth  # FT済GDino (1.2GB)
│       └── cfg_odvg.py
├── docs/                   # ドキュメント
└── tests/                  # テスト
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

# GroundingDino (性器検出)
models/gdino/dino_local_ft_ep4_best.pth  # ローカルFT済み (epoch 4)
models/gdino/cfg_odvg.py                  # ODVG設定ファイル
```

## 精度指標 (2026-05-04)

### 顔パイプライン (testset_2603, n=2837)

| Config | Coverage | Precision | IoU |
|--------|----------|-----------|-----|
| 施策D hybrid (conf=0.3) | 0.6491 | 0.8024 | 0.5792 |
| Production (conf=0.5) | 0.5482 | 0.8463 | 0.5130 |

### 性器パイプライン (testset_2603, n=2353)

| Config | Coverage | Precision | IoU |
|--------|----------|-----------|-----|
| FT ep4_best (box_thr=0.20) | 0.484 | 0.480 | 0.326 |

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
