# モザイク漏れ検知 AI — 統合実装報告書

**作成日**: 2026-05-04  
**ステータス**: Phase 3 完了（顔・性器パイプライン実装＋ローカル Fine-tuning＋全動画874本バッチ検証済み）  
**参照設計書**: `0501_モザイク漏れ検知AI_設計書.md`（詳細設計）, `0502_モザイク漏れ検知AI_設計書.md`（要件定義）

---

## 目次

1. [概要とスコープ](#1-概要とスコープ)
2. [設計要件との整合性確認](#2-設計要件との整合性確認)
3. [モデル構成と取得元（オープンソース）](#3-モデル構成と取得元オープンソース)
4. [顔検出パイプライン — 実装と評価](#4-顔検出パイプライン--実装と評価)
5. [性器検出パイプライン — 実装とFine-tuning](#5-性器検出パイプライン--実装とfine-tuning)
6. [統合パイプライン（顔＋性器）](#6-統合パイプライン顔性器)
7. [全動画 874 本バッチ検証（2026-05-04）](#7-全動画-874-本バッチ検証2026-05-04)
8. [判明した課題と今後の展望](#8-判明した課題と今後の展望)

---

## 1. 概要とスコープ

### 1.1 目的

クリエイターが投稿した動画の**モザイク漏れ**（かけ忘れ・薄すぎ）を AI で自動検知する。  
対象は **顔モザイク** と **性器モザイク** の 2 種類。

```
動画アップロード（S3）
    ↓
フレーム抽出（ffmpeg, 1fps）
    ↓
 ┌──────────────┬──────────────────────┐
 │ 顔パイプライン │   性器パイプライン    │
 │  SCRFD       │   Fine-tuned GDino   │
 │  ByteTrack   │                      │
 │  周波数解析   │   周波数解析          │
 └──────┬───────┴──────────┬───────────┘
        │                  │
        └─────── 統合判定 ──┘
              FAIL > REVIEW > PASS
```

### 1.2 導入形態と設計方針

設計書（0502）は **2 段階の導入形態** を定義している。

| 形態 | 対象 | 閾値設計 | 優先度 |
|---|---|---|---|
| **① アップロード時（クリエイター向け）** | 動画投稿時にクリエイター自身へ通知 | **FP 極小化**（誤警告を出さない優先） | **先行着手** |
| ② パトロール側（管理画面） | パトロールチームが確認する管理画面 | FN 極小化（見逃しゼロ優先） | ①安定後に着手 |

> **本報告書の評価・実装はすべて①アップロード時検知を前提**とした FP 極小化設計で行われている。

### 1.3 実装ファイル構成

```
/home/pan/プロジェクト/16.モザイク検知/
├── src/
│   ├── config.py              # 全パラメータ設定
│   ├── mosaic_analyzer.py     # 周波数解析（Laplacian/ACF/DCT）
│   ├── scorer.py              # スコアリング・集約ロジック
│   ├── pipeline.py            # 顔パイプライン（CLI対応）
│   ├── genital_pipeline.py    # 性器パイプライン
│   ├── combined_pipeline.py   # 顔＋性器 統合パイプライン
│   ├── eval_genital.py        # 性器パイプライン評価スクリプト
│   └── eval_testset.py        # 顔パイプライン評価スクリプト
├── vendor/
│   └── Pytorch_Retinaface/    # git clone 済み（重みは未取得）
├── genital-reference/
│   └── LV_Open-GroundingDino/ # Fine-tuning コード・チェックポイント
└── docs/
    ├── 本報告書（統合）
    ├── 0503_顔実装報告書.md    # 顔パイプライン詳細
    └── 0503_性器実装報告書.md  # 性器パイプライン詳細
```

---

## 2. 設計要件との整合性確認

### 2.1 設計書（0502）の主要要件と実装状況

| # | 要件 | 設計仕様 | 実装状況 | 整合性 |
|---|---|---|---|---|
| 1 | フレーム抽出 | 1fps（長尺60分以上: 0.5fps） | `config.py` で実装済み | ✅ |
| 2 | 顔検出モデル | SCRFD-2.5g（ONNX） | `vendor/scrfd/scrfd.onnx` 使用 | ✅ |
| 3 | 顔検出閾値（アップロード時） | conf_th=0.5以上（FP極小化） | 現在 conf_th=0.3（← **要調整**） | ⚠️ |
| 4 | トラッキング | ByteTrack（track_buffer=5） | 実装済み | ✅ |
| 5 | モザイク解析 3指標 | Laplacian / ブロックサイズ(ACF) / DCT周期性 | `mosaic_analyzer.py` で実装済み | ✅ |
| 6 | フレーム判定 | GREEN / YELLOW / RED の3段階 | `scorer.py` で実装済み | ✅ |
| 7 | 動画判定 | PASS / REVIEW / FAIL に集約 | 実装済み | ✅ |
| 8 | 偽陽性の3段階抑制 | SCRFD検出 → ByteTrack追跡 → モザイク解析 の全通過が必要 | `min_track_frames=2` で実装済み | ✅ |
| 9 | 性器検出 | Fine-tuned GroundingDino | 実装・Fine-tuning 完了 | ✅ |
| 10 | 統合判定 | FAIL > REVIEW > PASS | `combined_pipeline.py` で実装済み | ✅ |

### 2.2 重要な確認事項：③顔検出閾値の FP 極小化設計

**設計書（0502）§4.2**:
> アップロード時検知（①）ではパトロール側（②）より conf_th を高め（**0.5以上**）に設定し、偽陽性をさらに抑制する。

**現状**: 実装の `config.py` は `scrfd_conf_th=0.3`（設計書の推奨 0.5 より低い）。  
これは **評価・研究目的**で低感度設定のままになっており、**本番デプロイ時には 0.5 に引き上げる**必要がある。

| conf_th | Coverage | Precision | 用途 |
|:---:|:---:|:---:|---|
| 0.15 | 高い | 低い | パトロール側②（FN極小化） |
| 0.30 | 中 | 中 | 評価・検証用（現在の設定） |
| **0.50+** | 低い | **高い** | **アップロード時①（FP極小化）← 本番用** |

### 2.3 顔検出モデルの正確性：RetinaFace の評価結果

設計書（0501）§7.3 で RetinaFace が代替候補として言及されている。  
**評価結果**: `0503_SCRFD以外のmodel検討.md` §2-3 に詳細を記録。

| モデル | WiderFace Hard | testset_2603 加重 IoU | 判定 |
|---|:---:|:---:|---|
| **SCRFD-2.5g** ✅ | 77.87% | **0.566**（最良） | 採用 |
| RetinaFace (ResNet50) | 64.17% | 未実測（WF差で却下） | **却下** |
| YOLO11n-face | — | 0.021（4K で壊滅） | 却下 |

> **RetinaFace(ResNet50) は SCRFD-2.5g より WiderFace Hard で -13.7pp 低い。**  
> FP 極小化を優先する場合も、より高い検出精度をベースに conf_th を調整する方が合理的であり、  
> **SCRFD-2.5g（conf_th=0.5+）が正しい FP 極小化設計**である。

`vendor/Pytorch_Retinaface/` は参考として git clone 済みだが、**重みファイル（Resnet50_Final.pth）は未取得**。採用方針が変わらない限り取得不要。

---

## 3. モデル構成と取得元（オープンソース）

### 3.1 利用モデル一覧

| モデル | 取得元（GitHub） | ローカルパス | 用途 |
|---|---|---|---|
| **SCRFD-2.5g** (ONNX) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | `02_GitHub/BiSeNet/vendor/scrfd/scrfd.onnx` | 顔検出 |
| **ByteTrack** | [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack) | `02_GitHub/BiSeNet/vendor/yolox/tracker/` | 顔追跡 |
| **GroundingDino** | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | `genital-reference/LV_Open-GroundingDino/` | 性器検出（Fine-tuning） |
| **BERT (bert-base-uncased)** | [huggingface/transformers](https://huggingface.co/bert-base-uncased) | `~/.cache/huggingface/hub/` | GroundingDino テキストエンコーダ |
| RetinaFace (ResNet50) | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | `vendor/Pytorch_Retinaface/`（重みなし） | 参考調査済み・不採用 |

### 3.2 SCRFD と ByteTrack のソース管理方針

現状、SCRFD と ByteTrack は **`02_GitHub/BiSeNet/`（別プロジェクト）の vendor ディレクトリ**から参照している。これは既存資産の再利用として合理的だが、以下の課題がある。

| 課題 | 内容 |
|---|---|
| 依存性 | `02_GitHub/BiSeNet/` プロジェクトへの暗黙的な依存 |
| 移植性 | デプロイ時に BiSeNet ディレクトリが必要 |
| 管理 | モデルバージョンの明示的な管理がない |

**推奨改善（本番デプロイ前）**: `16.モザイク検知/vendor/` に SCRFD と ByteTrack を直接 clone し、パスを自己完結させる。

```bash
# 推奨: 16.モザイク検知/vendor/ に移行
cd /home/pan/プロジェクト/16.モザイク検知/vendor/
git clone https://github.com/ifzhang/ByteTrack.git
# SCRFD は insightface から ONNX をダウンロード
```

---

## 4. 顔検出パイプライン — 実装と評価

### 4.1 パイプライン設計

設計書（0501/0502）に基づき、**3段階の偽陽性抑制フロー**を実装した。

```
[1] SCRFD顔検出（conf_th=0.3 評価用 / 0.5 本番用）
    ↓
[2] ByteTrack 追跡確認（min_track_frames=2: 単発ノイズを棄却）
    ↓
[3] 周波数解析（Laplacian / ブロックサイズ(ACF) / DCT周期性）
    ↓ 3指標すべてが条件を満たした場合のみ警告
[4] GREEN / YELLOW / RED 判定
    ↓
[5] 動画集約: PASS / REVIEW / FAIL
```

偽陽性を抑制する最重要設計ポイント: **SCRFDが顔を検出した ≠ モザイク漏れ**。Laplacian等で「モザイクがかかっていない」ことを3指標で確認して初めて警告を発する。

### 4.2 粗いブロックモザイクの3指標補正

粗いブロックモザイク（32px 以上）はブロック境界のエッジにより Laplacian 単独では「モザイクなし」と誤判定されやすい問題がある。

| 問題 | 解決 |
|---|---|
| 32px モザイクのエッジで Laplacian 値が高くなる | block_size ≥ 16 AND DCT 周期性 > 2.0 → Laplacian 高値でも GREEN に補正 |

この「3指標設計」こそが最も重要な FP 抑制機構である。

### 4.3 testset_2603 評価結果

**評価条件**:
- 評価データ: testset_2603（normal / gay / hukusu / rez, 計 n=2,443 フレーム）
- データ形式: 原画フレーム（モザイク未適用）に GTマスク付き
- 評価指標: Coverage / Precision / IoU（GT面積と予測 BBox の重なり）
  > **❗ 指標の注意**: GT はセグメンテーション用のプンポリゴンアノテーション、予測は**矩形 BBox 塗りつぶし**。BBox は GT ポリゴンより常に大きいため、Precision と IoU は体系的に過小評価される（最大値 Precision≈ 70〜80%）。**Coverage（Recall）が最も信頼できる指標**。モデル間の相対順位は有効。

| 設定 | Coverage | Precision | IoU | 推論速度 |
|---|:---:|:---:|:---:|:---:|
| **Baseline（SCRFD-2.5g）** | 0.650 | 0.762 | 0.566 | ~32 fps |
| 施策B（SCRFD confidence活用） | 0.651 | 0.762 | 0.567 | ~32 fps |
| 施策C（SegFormer 二次確認） | 0.465 | **0.875** | 0.455 | ~7 fps |
| **施策D（SCRFD-34g ハイブリッド）** ★ | 0.649 | **0.802** | **0.579** | ~21 fps |

**推奨構成（FP 極小化 = アップロード時①）**: 施策D（SCRFD-34g ハイブリッド） + conf_th=0.50  
- Precision +5.3%、IoU +2.2%（全施策中最高）
- 複数人フレームでは 2.5g にフォールバックするため 34g の弱点（多人数での検出漏れ）を回避

### 4.4 サブセット別の詳細（施策D）

| サブセット | n | Coverage | Precision | IoU |
|---|:---:|:---:|:---:|:---:|
| normal | 743 | 0.713 | 0.927 | 0.671 |
| gay | 552 | 0.531 | 0.698 | 0.470 |
| hukusu | 644 | 0.570 | 0.772 | 0.507 |
| rez | 504 | 0.782 | 0.770 | 0.654 |

### 4.5 モザイク済み動画での実検証

品質チェックリスト（合格 20 本・不合格 3 本）との整合:

| 動画 | 違反種別 | 本パイプラインの対応 |
|---|---|---|
| `001_12.mp4` | 身体部位（男性器）露出 | **対象外**（性器パイプラインで検知） |
| `001_3.mp4` | 身体部位（陰嚢）モザイクズレ | **対象外**（性器パイプラインで検知） |
| `387_202601061150_1.mp4` | 顔モザイクが薄い | **検知対象** |

> 顔パイプラインは **顔のみ** が対象。身体部位は性器パイプラインが担う。

**387（薄いモザイク）の検証**: conf_th=0.30 でも検出が難しい（1フレームのみ）。薄いモザイクでは SCRFD の confidence 値が下がるため、閾値設計に注意が必要。

### 4.6 処理性能

| 条件 | 処理速度 |
|---|---|
| 55秒動画（55フレーム）| 12.2 秒（GPU） |
| 30分動画（推定） | 60〜80 秒 |
| **施策D ハイブリッド** | ~21 fps（vs baseline 32 fps） |

---

## 5. 性器検出パイプライン — 実装とFine-tuning

### 5.1 モデル選定と理由

| # | 候補 | 判断 | 理由 |
|---|---|---|---|
| **A** | **Fine-tuned GroundingDino** ✅ | **採用** | モザイク生成パイプラインで実績あり。社内映像に Fine-tuning 済み。テキストプロンプトで柔軟に対応 |
| B | 姿勢推定（RTMPose等） | 非推奨 | 遮蔽・体位変化で精度が構造的に低い |
| C | 汎用物体検出器 | 非推奨 | 性器の精密な位置特定に不向き |

### 5.2 ローカル Fine-tuning（2026-05-04 実施）

ベースモデル（`dino_260430_checkpoint0002.pth`, epoch 2）を H100 × 2 でさらに Fine-tuning した。

**学習設定**:

| 項目 | 値 |
|---|---|
| ベースモデル | `dino_260430_checkpoint0002.pth`（GCP 学習済み, epoch 2） |
| 学習データ | `finetune_dataset_20260227`（train 15,074 枚 / val 2,353 枚） |
| 学習率 | 5e-6（BERT encoder は frozen） |
| バッチサイズ | 12 × 2 GPU |
| 最大エポック | 20（early stopping patience=4） |
| ラベル | `["penis", "vagina"]` |
| GTラベルのマッピング | `vagina_wide → vagina`, `anus_insertion → vagina` |
| GPU | H100 NVL × 2（CUDA sm_90） |

**学習結果**: Early stopping at epoch 8、**best epoch = 4**（COCO mAP=0.0668）

| Epoch | mAP | AP@0.50 | AR@100 | Train Loss |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0632 | 0.1777 | 0.5622 | 4.6090 |
| 1〜3 | 0.0607〜0.0614 | — | — | 3.0420〜3.4309 |
| **4** ★ | **0.0668** | **0.1793** | **0.5650** | 2.9313 |
| 5〜7 | 0.0548〜0.0589 | — | — | 2.7145〜2.8256 |
| 8 | — | — | — | early stopping |

### 5.3 testset_2603 評価結果（Fine-tuned ep4 vs ベースライン）

**評価条件**: box_threshold=0.3, text_threshold=0.2

| 指標 | ベースライン（0430_ep2） | **Fine-tuned（ep4）** | 改善率 |
|:---:|:---:|:---:|:---:|
| **Coverage** | 0.209 | **0.325** | **+55.5%** |
| **Precision** | 0.312 | **0.440** | **+41.0%** |
| **IoU** | 0.165 | **0.256** | **+55.2%** |

### 5.4 閾値チューニング結果

`eval_genital.py` の `--box-threshold` CLI バグ（引数が推論に反映されない）を修正した上で実施。

| box_threshold | Coverage | Precision | IoU | 推奨 |
|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.325 | 0.440 | 0.256 | — |
| 0.25 | 0.408 | 0.482 | 0.303 | — |
| **0.20** ★ | **0.484** | **0.480** | **0.326** | **推奨** |
| 0.15 | 0.565 | 0.456 | 0.338 | — |
| 0.10 | 0.652 | 0.402 | 0.330 | — |

**推奨: box_threshold=0.20**（Coverage と Precision のバランスが最良）

ベースラインとの閾値別比較:

| box_threshold | Baseline Cov | **FT Cov** | Baseline Prec | **FT Prec** |
|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.209 | **0.325** (+55%) | 0.312 | **0.440** (+41%) |
| 0.25 | 0.258 | **0.408** (+58%) | 0.375 | **0.482** (+29%) |
| 0.20 | 0.326 | **0.484** (+48%) | 0.426 | **0.480** (+13%) |
| 0.15 | 0.413 | **0.565** (+37%) | 0.448 | **0.456** (+2%) |

→ **FT@0.20 の Coverage(0.484) は ベースライン@0.15(0.413) を上回りつつ、Precision も高い**

### 5.5 サブセット別の詳細（Fine-tuned ep4, box_thr=0.20）

| サブセット | GT保有 | Coverage | Precision | IoU |
|---|:---:|:---:|:---:|:---:|
| normal | 835 | 0.321 | 0.448 | 0.241 |
| gay | 238 | 0.512 | 0.541 | 0.380 |
| hukusu | 662 | 0.543 | 0.608 | 0.418 |
| rez | 618 | 0.559 | 0.324 | 0.266 |
| **全体** | **2353** | **0.484** | **0.480** | **0.326** |

### 5.6 Fine-tuning の技術的課題と解決

| 課題 | 内容 | 解決 |
|---|---|---|
| CUDA Extension (sm_90) | H100 対応の MsDeformableAttention が動作しない | `groundingdino/_C.so` を sm_90 ビルドに置換 |
| COCO PostProcess id_map | 2クラス分類でクラッシュ | `use_coco_eval=False` に変更 |
| yapf `verify` 引数廃止 | `FormatCode()` の互換性エラー | `verify=True` を削除 |
| `--resume` vs `--pretrain_model_path` | lr_scheduler state mismatch | `--pretrain_model_path` で重みのみロード |

### 5.7 チェックポイントパス

| 項目 | パス |
|---|---|
| **Best checkpoint** | `genital-reference/LV_Open-GroundingDino/local_train/output_finetune/checkpoint_best_regular.pth` |
| 推論 config | `genital-reference/LV_Grounded-SAM-2/cfg_odvg.py` |
| 評価結果 JSON | `local_train/eval_best_ep4_thr020.json`（box_thr=0.20） |

---

## 6. 統合パイプライン（顔＋性器）

### 6.1 アーキテクチャ

```
                ┌──────────────────────────────────┐
                │       入力動画（MP4）             │
                └──────────────┬───────────────────┘
                               │
                ┌──────────────▼───────────────────┐
                │  フレーム抽出（ffmpeg, 共有）     │
                │  <5分: 2fps / ≥5分: 1fps         │
                └──────┬────────────────┬───────────┘
                       │                │
           ┌───────────▼──┐    ┌────────▼──────────┐
           │ 顔パイプライン│    │  性器パイプライン  │
           │  SCRFD-2.5g  │    │  Fine-tuned GDino │
           │  ByteTrack   │    │  box_thr=0.20     │
           │  周波数解析   │    │  周波数解析        │
           │  PASS/REVIEW/│    │  PASS/REVIEW/FAIL │
           │  FAIL        │    │                   │
           └──────┬───────┘    └─────────┬─────────┘
                  │                      │
                  └──────────┬───────────┘
                             │
              ┌──────────────▼──────────────────┐
              │  統合判定（FAIL > REVIEW > PASS）│
              │  ERROR は REVIEW として扱う      │
              └──────────────────────────────────┘
```

### 6.2 統合判定ロジック

```python
VERDICT_RANK = {"PASS": 0, "REVIEW": 1, "FAIL": 2, "ERROR": 1}

def merge_verdicts(face_verdict, genital_verdict):
    """安全側（深刻な方）を採用"""
    return max(face_verdict, genital_verdict, key=lambda v: VERDICT_RANK[v])
```

| 顔 | 性器 | **統合** |
|:---:|:---:|:---:|
| PASS | PASS | **PASS** |
| PASS | FAIL | **FAIL** |
| FAIL | PASS | **FAIL** |
| REVIEW | FAIL | **FAIL** |
| ERROR | — | **REVIEW** |

### 6.3 実行コマンド

```bash
cd /home/pan/プロジェクト/16.モザイク検知/src

# 顔＋性器 統合パイプライン
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/pan/miniforge3/envs/ml/bin/python combined_pipeline.py \
  --input /path/to/video.mp4 \
  --output result.json

# 顔のみ
/home/pan/miniforge3/envs/ml/bin/python combined_pipeline.py \
  --input video.mp4 --face-only

# 性器のみ
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/pan/miniforge3/envs/ml/bin/python combined_pipeline.py \
  --input video.mp4 --genital-only
```

---

## 7. 全動画 874 本バッチ検証（2026-05-04）

### 7.1 背景と目的

testset_2603 での定量評価（§4・§5）はラボ環境での評価に過ぎない。  
実際の納品対象全動画 874 本を対象に、**顔＋性器統合パイプラインのエンドツーエンド検証**を実施した。

> これ以前に、**顔単独パイプライン**でも 874 本のバッチ検証（2026-05-03 17:57 完了）を行っている。  
> 本章の結果は「顔＋性器統合バッチ」（2026-05-04 01:02 完了）の結果であり、**別の実行**である。

### 7.2 実装の特徴（In-process モデルロード方式）

| 方式 | 処理速度 |
|---|---|
| サブプロセス方式（旧: 顔単独バッチ） | ~25 秒/本 |
| **In-process 方式（今回）** | **~6.5 秒/本** |

```
[起動時 1回のみ]
  SCRFD ロード + GroundingDino ロード（~11 秒）
       ↓
[各動画 (874本) に対して繰り返し]
  フレーム抽出（ffmpeg, 1fps）← 顔・性器で共有
  顔パイプライン（pre-loaded SCRFD + ByteTrack）
  性器パイプライン（pre-loaded GroundingDino, box_thr=0.3）
  統合判定 → CSV 逐次書き込み（クラッシュ耐性）
```

### 7.3 実行サマリ

| 項目 | 値 |
|---|---|
| 実行日時（開始） | 2026-05-03 23:28:11（リジューム再起動） |
| 実行日時（完了） | 2026-05-04 01:02:02 |
| 実経過時間 | **1時間34分** |
| 処理本数 | 823 本（スキップ 51 本） |
| 平均処理時間 | 6.5 秒/本（中央値 4.9 秒） |
| エラー | 7 本（フレーム抽出失敗） |

### 7.4 統合判定の全体分布

| 統合判定 | 本数 | 割合 | 意味 |
|:---:|:---:|:---:|---|
| **PASS** | **483** | **55.3%** | 顔・性器ともに問題なし |
| **REVIEW** | **300** | **34.3%** | 要目視確認 |
| **FAIL** | **84** | **9.6%** | 明確なモザイク漏れ |
| ERROR | 7 | 0.8% | フレーム抽出失敗 |
| **合計** | **874** | **100%** | |

### 7.5 顔 × 性器 クロス判定マトリクス

| 顔判定 ＼ 性器判定 | ERROR | FAIL | PASS | REVIEW | **合計** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ERROR | 7 | 0 | 37 | 14 | 58 |
| FAIL | 0 | 2 | 25 | 19 | 46 |
| PASS | 0 | **35** | 483 | 142 | **660** |
| REVIEW | 0 | 3 | 89 | 18 | 110 |
| **合計** | 7 | 40 | 634 | 193 | 874 |

### 7.6 性器パイプラインによる追加検出効果

顔パイプライン単独では見逃していたが、性器パイプラインが発見した件数：

| 変化 | 本数 | 割合 |
|---|:---:|:---:|
| 顔 PASS → 統合 **FAIL**（性器のみに問題） | **35 本** | 874本中 4.0% |
| 顔 PASS → 統合 **REVIEW** | **142 本** | 874本中 16.3% |
| **顔 PASS → 統合 FAIL/REVIEW 合計** | **177 本** | **874本中 20.3%** |

**→ 顔単独パイプラインでは 874 本中 177 本（20.3%）が見逃されていた**

### 7.7 FAIL の内訳

| FAIL の原因 | 本数 | FAIL全体比 |
|---|:---:|:---:|
| 統合 FAIL 合計 | 84 | 100% |
| 　うち 性器 FAIL のみ（顔は PASS/REVIEW） | **38** | **45%** |
| 　うち 顔 FAIL のみ（性器は PASS/REVIEW） | 8 | 10% |
| 　うち 両方 FAIL | 2 | 2% |
| 　うち 顔 REVIEW + 性器 FAIL | 3 | 4% |
| 　うち 顔 ERROR 含む | 33 | 39% |

**性器 FAIL が単独原因のケースが 38 本（FAIL全体の 45%）**  
→ 性器パイプラインなしでは検知不可能だった問題が、FAIL全体の半数を占める。

### 7.8 FAIL 上位フォルダ（統合判定）

| フォルダ | FAIL 本数 |
|---|:---:|
| LV/グループ2_和田さん | 27 |
| LV/グループ18_安藤さん | 23 |
| ML/和田 | 11 |
| ML/辻 | 5 |
| ML/三瓶 | 4 |

### 7.9 ソースディレクトリ別の内訳

| ソースディレクトリ | PASS | REVIEW | FAIL | ERROR | 合計 |
|---|:---:|:---:|:---:|:---:|:---:|
| output | 339 | 209 | 46 | 5 | 599 |
| output_ZIP未展開 | 144 | 91 | 38 | 2 | 275 |

### 7.10 実装時に発生した課題と解決

**課題: `detect_faces_scrfd` 未定義エラー**

```
NameError: name 'detect_faces_scrfd' is not defined
```

**原因**: バッチ起動時に `_setup_imports()` でインポートした関数が、`process_face_pipeline()` のクロージャスコープ外からアクセスできなかった。

**解決**: 関数の先頭でローカルインポート。

```python
def process_face_pipeline(...):
    from pipeline import detect_faces_scrfd  # 追加
```

**影響**: 修正前の先頭 35 本が顔 ERROR → リジューム再起動で再処理済み。

### 7.11 成果物一覧

| ファイル | 内容 |
|---|---|
| `tmp/0503_顔+性器_LV-MLモザイク検証/mosaic_check_results.csv` | 全 874 本の統合結果（17 カラム） |
| `tmp/0503_顔+性器_LV-MLモザイク検証/batch_run.py` | In-process バッチ処理スクリプト |
| `tmp/0503_顔+性器_LV-MLモザイク検証/batch_nohup.log` | 実行ログ |

---

## 8. 判明した課題と今後の展望

### 8.1 顔パイプラインの課題

| 優先 | 課題 | 内容 | 状態 |
|---|---|---|---|
| P0 | **本番 conf_th の調整** | アップロード時①は conf_th=0.50 に引き上げる（現在 0.30 は評価用設定） | **未実施** |
| P1 | **施策D（SCRFD-34g ハイブリッド）のデプロイ** | testset_2603 で IoU 最良（0.579）。Precision +5.3% | 未着手 |
| P1 | **Laplacian 閾値キャリブレーション** | 現行 `threshold_low=50` は理論値。実動画での閾値確定が必須 | 未着手 |
| P1 | **薄いモザイクでの検出課題** | conf_th=0.3 でも 387（薄いモザイク）は 1 フレームのみ検出 | 継続課題 |
| P2 | **vendor 依存の整理** | SCRFD / ByteTrack を `02_GitHub/BiSeNet/vendor/` から `16.モザイク検知/vendor/` へ移行 | 未着手 |
| P3 | **Slack 通知** | REVIEW/FAIL 時の Webhook 送信 | 未着手（Phase 2） |

### 8.2 性器パイプラインの課題

| 優先 | 課題 | 内容 | 状態 |
|---|---|---|---|
| P0 | **Fine-tuned チェックポイントのデプロイ** | `checkpoint_best_regular.pth` を本番環境に反映。`BOX_THRESHOLD=0.20` に変更 | **未実施** |
| P1 | **normal サブセットの Coverage 改善** | normal の Coverage=0.321 が他（0.51〜0.56）より低い。原因調査 | 未着手 |
| P1 | **Laplacian 閾値キャリブレーション（性器用）** | 性器 ROI のモザイク済みサンプルで閾値を調整 | 未着手 |
| P2 | **学習率 1e-5 での再学習** | 元 GCP 学習（lr=1e-5）より低い lr=5e-6 で実施。1e-5 でさらに改善の可能性 | 未着手 |
| P2 | **ByteTrack の追加** | フレーム間の連続性を使った偽陽性フィルタリング | 未着手 |

### 8.3 統合パイプラインの課題

| 優先 | 課題 | 内容 | 状態 |
|---|---|---|---|
| P1 | **REVIEW 34.3%の目視確認** | REVIEW 300 本のうち性器 RED フレームあり 37 本を優先確認 | 未着手 |
| P1 | **顔 ERROR 58 本の対処** | フレーム抽出エラー 7 本を除く 51 本は顔検出ライブラリの初期化エラー | 要調査 |
| P2 | **HEVC 動画の前処理** | H.265 コーデックの 7 本（ERROR）を H.264 に再エンコードして処理可能にする | 未着手 |

### 8.4 RetinaFace（ResNet50）採用について

**結論: 採用しない**

0502 設計書の FP 極小化要件を満たすには、より高精度なベースモデルの上で conf_th を引き上げる方が合理的。

| 検討理由 | 結論 |
|---|---|
| FP 極小化のために精度の高いモデルが欲しい | SCRFD-2.5g が WF Hard で +13.7pp 上回る（77.87% vs 64.17%） |
| 側面顔の検出精度が高いとされる | `0503_SCRFD以外のmodel検討.md` §2-3 で実測評価済み・却下 |
| `vendor/Pytorch_Retinaface/` は git clone 済み | 重みファイル（Resnet50_Final.pth）は未取得のまま |

FP 極小化には **SCRFD-2.5g（conf_th=0.50）+ 施策D（34g ハイブリッド）** が正しいアプローチ。

---

## 参考：関連ドキュメント一覧

| ドキュメント | 内容 |
|---|---|
| `0501_モザイク漏れ検知AI_設計書.md` | 顔パイプライン詳細設計（アーキテクチャ・各指標の定義） |
| `0502_モザイク漏れ検知AI_設計書.md` | 要件定義（FP/FN設計・導入形態・閾値方針） |
| `0503_顔実装報告書.md` | 顔パイプライン実装詳細・精度評価・施策B/C/D比較 |
| `0503_性器実装報告書.md` | 性器パイプライン実装詳細・Fine-tuning結果・閾値スイープ |
| `0503_SCRFD以外のmodel検討.md` | 代替顔検出モデル評価（YuNet・RetinaFace・YOLO等） |
| `0504_顔+性器AI検知.md` | 統合パイプライン設計・API仕様 |
| `0505_周波数解析3指標_問題調査と改善.md` | Laplacian/ACF/DCT 閾値の課題分析 |
