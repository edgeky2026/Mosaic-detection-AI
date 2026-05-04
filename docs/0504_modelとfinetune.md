# モザイク検知AI — モデル選定とファインチューニング

> 作成日: 2026-05-04  
> 目的: 顔/性器モザイク検知で使用するモデルの技術仕様と、ファインチューニングの必要性・実施内容を整理する

---

## 1. 使用モデル一覧

| 対象 | モデル | 役割 | FT有無 | 理由 |
|---|---|---|:---:|---|
| 顔 | SCRFD-2.5g | 顔領域のBBox検出 | **なし** | 汎用顔検出器として十分な性能。追加学習は設計原理に反する |
| 顔 (補助) | SCRFD-34g | 高精度フォールバック | **なし** | 同上（単体顔フレーム時のみ使用） |
| 性器 | GroundingDINO (Swin-T) | 性器領域のBBox検出 | **あり** | ドメイン固有（モザイク性器）のため事前学習モデルでは検出不能 |

---

## 2. SCRFD（Sample and Computation Redistribution for Face Detection）

### 2.1 モデル概要

SCRFD は InsightFace チームが開発した軽量・高速な顔検出モデルである。FPN（Feature Pyramid Network）ベースのアンカーフリー検出器で、計算量と精度のバランスを最適化している。

| 項目 | SCRFD-2.5g | SCRFD-34g |
|---|---|---|
| 計算量 | 2.5 GFLOPs | 34 GFLOPs |
| モデルサイズ | 3.3 MB (ONNX) | 39 MB (ONNX) |
| 推論速度 | ~15 ms/frame (GPU) | ~40 ms/frame (GPU) |
| 出力 | BBox (x1,y1,x2,y2) + confidence + 5 keypoints |
| 入力 | 640×640 (自動リサイズ) |
| ランタイム | ONNX Runtime (CUDA/CPU) |

### 2.2 本システムでの使い方

```
入力フレーム → SCRFD推論 → BBox + confidence score
                                    ↓
                            conf ≥ conf_th なら「顔検出」
                                    ↓
                            BBox 領域を切り出し → 周波数解析へ
```

**設計原理**: 「SCRFD が顔を検出できる ＝ モザイクが不十分」という逆説的な前提に基づく。十分なモザイクは顔の特徴を破壊するため、顔検出器に検出されない。

### 2.3 なぜファインチューニングしないか

1. **設計原理に反する**: 本システムは「汎用顔検出器が反応する ＝ モザイク不足」を前提とする。モザイク画像でも検出できるように学習させると、正常なモザイク動画でも誤検出が頻発し、設計が破綻する
2. **十分な性能**: WiderFace で学習済みの SCRFD は、モザイクの薄い顔に対して conf=0.3-0.8 で適切に反応する
3. **BBox出力のため問題なし**: セグメンテーションではなく矩形検出なので、既存の学習データがそのまま有効

### 2.4 ハイブリッド運用（施策D）

| 条件 | 使用モデル | 理由 |
|---|---|---|
| 1フレームに顔1つ | SCRFD-34g | 高精度（Precision +5.3%） |
| 1フレームに顔2つ以上 | SCRFD-2.5g | 34g は2人目を抑圧する傾向があるためフォールバック |

---

## 3. GroundingDINO（Grounding Detection with INteractive Objects）

### 3.1 モデル概要

GroundingDINO は、テキストプロンプトで任意の物体を検出できるオープンセット物体検出モデルである。Vision Transformer（Swin-T）とテキストエンコーダ（BERT）を融合し、テキスト-画像間のクロスアテンションで物体を特定する。

| 項目 | 値 |
|---|---|
| バックボーン | Swin-Transformer Tiny |
| テキストエンコーダ | bert-base-uncased |
| 出力 | BBox (cx,cy,w,h → xyxy変換) + logit score + phrase |
| テキストプロンプト | `"vagina . penis ."` |
| モデルサイズ | 1.2 GB (PyTorch) |
| 推論速度 | 7-9 fps (H100), ~5 fps (T4) |
| CUDA要件 | Deformable Attention (sm_90 for H100) |

### 3.2 本システムでの使い方

```
入力フレーム → GroundingDINO推論（text="vagina . penis ."）
                        ↓
                BBox + confidence score
                        ↓
            score ≥ box_threshold(0.20) なら「性器検出」
                        ↓
            BBox 領域を切り出し → 周波数解析へ
```

### 3.3 なぜファインチューニングが必要か

| 理由 | 詳細 |
|---|---|
| **ドメインギャップ** | 事前学習データにモザイク加工済み性器画像がほぼ含まれない |
| **事前学習モデルの性能不足** | 汎用モデルではCoverage≈0.1未満で実用不可能 |
| **テキストプロンプトの限界** | Open-set の利点はあるが、ドメイン固有の視覚パターンには適応が必要 |
| **BBOX出力のため学習可能** | セグメンテーションモデルではないので、矩形アノテーションで学習できる |

---

## 4. ファインチューニングの詳細（性器GroundingDINO）

### 4.1 アノテーションデータの変換フロー

```
LabelMe JSON（ポリゴンアノテーション: vagina_wide, penis, anus_insertion）
        ↓  convert_local_data.py
ポリゴンの外接矩形 → BBox (x1, y1, x2, y2) に変換
        ↓
ラベル正規化: vagina_wide → vagina, anus_insertion → vagina
        ↓
ODVG 形式（.jsonl）  ← GroundingDINO の学習フォーマット
        ↓
GroundingDINO fine-tuning
```

**重要**: LabelMe のセグメンテーション用ポリゴンは、BBox に変換して学習に使用する。セグメンテーション情報は学習に一切使われない。GroundingDINO は常にBBox出力である。

### 4.2 学習設定

| 項目 | 値 |
|---|---|
| 学習環境 | H100 NVL × 2 (95 GB VRAM each) |
| 学習データ | finetune_dataset_20260227（train 15,074枚 / val 2,353枚） |
| ラベル | `["penis", "vagina"]` (ID 0/1) |
| 学習率 | 5e-6 |
| バッチサイズ | 12 × 2 GPU |
| エポック数 | 20（early stopping at epoch 8, best = epoch 4） |
| BERT | **凍結**（テキストエンコーダは学習しない） |
| 初期化 | `--pretrain_model_path`（重みのみ、optimizer 初期化） |
| 評価指標 | COCO mAP |

### 4.3 チェックポイント比較

| チェックポイント | 学習元 | epoch | Coverage | Precision | IoU |
|---|---|---:|:---:|:---:|:---:|
| dino_0331_checkpoint0004 | GCP | 4 | 0.175 | 0.248 | 0.136 |
| dino_260430_checkpoint0002 | GCP | 2 | 0.209 | 0.312 | 0.165 |
| **dino_local_ft_ep4_best** ✅ | ローカル H100 | 4 | **0.484** | **0.480** | **0.326** |

### 4.4 ファインチューニングの効果

| 指標 | FT前 (GCP ep2) | FT後 (local ep4) | 改善率 |
|---|:---:|:---:|:---:|
| Coverage | 0.209 | **0.484** | **+131%** |
| Precision | 0.312 | **0.480** | **+54%** |
| IoU | 0.165 | **0.326** | **+98%** |

### 4.5 残課題

- **FP=2,245件**: 96.8% が体部位（腕/脚/腹部等）の誤検出。スコア分布が TP と重なるため、閾値調整のみでは解消困難
- **normal サブセットの Coverage=0.321**: 他サブセット(0.51-0.56)より低い
- **改善案**: 追加学習データ拡充（body part negative mining）、後段分類器の追加

---

## 5. まとめ

| 観点 | 顔 (SCRFD) | 性器 (GroundingDINO) |
|---|---|---|
| モデル種別 | BBOX 検出器 | BBOX 検出器 |
| FTの必要性 | 不要（設計原理に反する） | 必須（ドメインギャップ大） |
| アノテーション | — | LabelMe polygon → BBox 変換 → ODVG |
| 学習データ量 | — | 15,074 枚 |
| 効果 | — | Coverage +131%, Precision +54% |
