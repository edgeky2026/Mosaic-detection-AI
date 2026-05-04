"""施策C: SegFormer二次確認レイヤー

SCRFD が顔を検出した ROI に対して Fine-tuned SegFormer を適用し、
実際に顔ピクセルが残存しているか確認する。

顔ピクセル割合が segformer_face_ratio_min を下回る場合は
SCRFD の誤検出（帽子・手など）とみなし、GREEN に降格する。
"""

import sys
import os
from typing import Optional, Tuple

import numpy as np

# Segformer vendor パスを追加
_SEGFORMER_PROJ = os.environ.get("SEGFORMER_PROJ", "/home/pan/プロジェクト/02_GitHub/Segformer")
if _SEGFORMER_PROJ not in sys.path:
    sys.path.insert(0, _SEGFORMER_PROJ)


class SegFormerChecker:
    """Fine-tuned SegFormer で顔ピクセルの存否を確認するクラス"""

    def __init__(self, model_path: str, face_classes: Tuple[int, ...], device: str = "cuda"):
        """
        Args:
            model_path: Fine-tuned SegFormer の重みディレクトリパス
            face_classes: 顔クラスのインデックス（CelebAMask-HQ: 1-12）
            device: 'cuda' または 'cpu'
        """
        from utils.segformer_parser import FacePartsSegformer

        self._model = FacePartsSegformer(
            local_model_path=model_path,
            device=device,
            use_fp16=(device == "cuda"),
        )
        self._face_classes = np.array(list(face_classes), dtype=np.int64)

    def face_pixel_ratio(self, roi_bgr: np.ndarray) -> float:
        """
        ROI（BGR画像）内の顔ピクセル割合を返す。

        Fine-tuned モデル（BinaryAggregationLoss）のため
        logsumexp(face_classes logits) > logit[0] で二値化する。

        Args:
            roi_bgr: (H, W, 3) uint8 BGR 画像

        Returns:
            face_pixels / total_pixels (0.0〜1.0)
        """
        if roi_bgr.size == 0:
            return 0.0

        # SegFormer は RGB 入力
        roi_rgb = roi_bgr[:, :, ::-1].copy()

        # 19クラス logits を取得
        raw_logits = self._model.predict(roi_rgb, return_logits=True)  # (19, H, W)

        # logsumexp による二値化（binary_inference）
        fc = self._face_classes
        logits_face = raw_logits[fc]  # (N_face, H, W)
        max_val = logits_face.max(axis=0, keepdims=True)  # (1, H, W)
        face_lse = (
            np.log(np.exp(logits_face - max_val).sum(axis=0)) + max_val[0]
        )  # (H, W)
        binary_mask = (face_lse > raw_logits[0]).astype(np.uint8)  # (H, W) 0/1

        total = binary_mask.size
        if total == 0:
            return 0.0
        return float(binary_mask.sum()) / total

    def get_face_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        ROI内の顔ピクセルマスク (0/1) を返す。
        評価スクリプト等から使用するユーティリティ。

        Returns:
            (H, W) uint8 マスク (1=顔, 0=背景)
        """
        if roi_bgr.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        roi_rgb = roi_bgr[:, :, ::-1].copy()
        raw_logits = self._model.predict(roi_rgb, return_logits=True)  # (19, H, W)

        fc = self._face_classes
        logits_face = raw_logits[fc]
        max_val = logits_face.max(axis=0, keepdims=True)
        face_lse = (
            np.log(np.exp(logits_face - max_val).sum(axis=0)) + max_val[0]
        )
        return (face_lse > raw_logits[0]).astype(np.uint8)
