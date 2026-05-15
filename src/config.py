"""モザイク漏れ検知 AI — 設定値"""

from dataclasses import dataclass, field
from typing import Tuple
import os

# リポジトリルート (src/ の親ディレクトリ)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_REPO_ROOT, "models")


@dataclass
class Config:
    """全パラメータを集約する設定クラス"""

    # --- フレーム抽出 ---
    fps_normal: float = 1.0          # 通常動画（60分未満）
    fps_long: float = 0.5            # 長尺動画（60分以上）
    long_video_threshold_min: float = 60.0  # 長尺動画の判定閾値（分）
    max_resolution: int = 1920       # 長辺の最大解像度
    jpeg_quality: int = 95
    scene_change_threshold: float = 0.3

    # --- SCRFD 顔検出 ---
    scrfd_model_path: str = os.path.join(_MODELS_DIR, "scrfd", "scrfd_2.5g.onnx")
    scrfd_conf_th: float = 0.50      # 本番用 FP極小化（設計書0502§4.2: アップロード時①は0.5以上）
    scrfd_nms_th: float = 0.65
    scrfd_max_det: int = 15
    scrfd_providers: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

    # --- ByteTrack ---
    track_thresh: float = 0.3
    track_buffer: int = 5            # 1fps抽出のため 5フレーム = 5秒間の補完
    match_thresh: float = 0.8
    tracker_frame_rate: int = 1      # 1fps抽出に合わせる

    # --- ROI ---
    roi_padding: int = 10            # bbox四辺に追加するパディング(px)
    roi_normalize_size: int = 64     # 解像度正規化の固定サイズ

    # --- Laplacian ---
    laplacian_threshold_low: float = 50.0    # 以下ならモザイク十分
    laplacian_threshold_high: float = 500.0  # 以上ならモザイクなし

    # --- ブロックサイズ推定 ---
    block_size_green: int = 16       # 以上ならGREEN
    block_size_yellow: int = 8       # 以上ならYELLOW

    # --- DCT周期性 ---
    periodicity_threshold: float = 2.0       # 以上ならブロックモザイクと判定
    periodicity_threshold_half: float = 1.5  # YELLOW判定用

    # =========================================================
    # 性器検出向け Laplacian 閾値（キャリブレーション結果反映）
    # =========================================================
    # testset_2603 の GT 性器 ROI 計測結果:
    #   原画 Laplacian p5=6.9, p25=18.3, median=35.1, p95=509.7
    # 顔 ROI と比較して性器 ROI はテクスチャが少なく Laplacian が低い傾向があるため
    # 顔用閾値（50/500）とは別に設定する。
    genital_laplacian_threshold_low: float = 15.0   # 以下ならモザイク十分（原画 p25≒18）
    genital_laplacian_threshold_high: float = 500.0  # 以上ならモザイクなし

    # GDino confidence boost（施策B 相当）
    # GDino スコアは SCRFD より低い（TP median=0.27, p95=0.64）ため閾値を下げる
    genital_confidence_yellow_th: float = 0.40
    genital_confidence_red_th: float = 0.55
    # 注意: クリーンな顔ROIのDCT-CV（自然テクスチャ由来）は ~1.36 程度。
    # 1.0 に設定するとクリーン顔がYELLOW(→本来はRED)に軟化されるため 1.5 に引き上げた。
    # 8px ブロックモザイクのDCT-CV は ~1.70 であり、1.5 で YELLOW判定が保たれる。

    # --- 動画集約 ---
    red_rate_fail: float = 0.05      # RED率がこれを超えたらFAIL
    consecutive_red_fail: int = 3    # 連続REDフレームがこれ以上でFAIL
    yellow_rate_review: float = 0.15 # YELLOW率がこれを超えたらREVIEW

    # --- ByteTrack安全策 ---
    min_track_frames: int = 2        # 追跡が確認された最小フレーム数（偽陽性棄却用）

    # =========================================================
    # 施策B: SCRFD confidence → 深刻度エスカレーション
    # =========================================================
    use_confidence_boost: bool = True    # デフォルト有効（自然ぼかし顔の偽GREEN対策）
    # 理由: 自然ぼかし/低解像度の顔でLaplacianが低くなりGREEN誤判定が生じる。
    # SCRFDが高信頼度で検出した顔は視覚特徴が残存しているためREDへ強制昇格させる。
    # 適切にモザイクがかかった顔のSCRFD信頼度は 0.3 未満（=検出されない）のため
    # 真のGREENケースはこの施策の影響を受けない。
    confidence_yellow_th: float = 0.70   # この信頼度以上でYELLOW以上に昇格
    confidence_red_th: float = 0.85      # この信頼度以上でRED方向に昇格

    # =========================================================
    # 施策C: SegFormer二次確認レイヤー
    # =========================================================
    use_segformer_confirmation: bool = False
    segformer_model_path: str = (
        os.path.join(_MODELS_DIR, "segformer", "segformer_v2_best")
    )
    # ファインチューニング済みモデルのface classes (CelebAMask-HQ class 1-12)
    segformer_face_classes: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    # ROI内の顔ピクセル割合がこれ以下 → GREEN降格（SCRFD誤検出とみなす）
    segformer_face_ratio_min: float = 0.05

    # =========================================================
    # 施策D: SCRFD-34g ハイブリッド検出
    # =========================================================
    use_scrfd34g_hybrid: bool = True   # 施策D採用済み: Precision+5.3%, IoU+2.3%
    scrfd_34g_model_path: str = (
        os.path.join(_MODELS_DIR, "scrfd", "scrfd_34g.onnx")
    )


