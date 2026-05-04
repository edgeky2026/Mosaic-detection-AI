"""
モザイク漏れ検知パイプライン — testset_2603 評価スクリプト

testset_2603 (normal / gay / hukusu / rez) の各フレームに対して
SCRFD 顔検出を実行し、GT ポリゴンマスクと比較して
Coverage / Precision / IoU を算出する。

評価設定（--config）:
  baseline   : 施策なし（現行パイプライン）
  施策B      : SCRFD confidence → 深刻度エスカレーション
  施策C      : SegFormer 二次確認レイヤー
  施策D      : SCRFD-34g ハイブリッド
  all        : 施策B+C+D 全適用

使用方法:
  /home/pan/miniforge3/envs/ml/bin/python eval_testset.py --config all
  /home/pan/miniforge3/envs/ml/bin/python eval_testset.py --config baseline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# --- パス設定 ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

# BiSeNet vendor (SCRFD / ByteTrack)
SCRFD_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
if SCRFD_VENDOR not in sys.path:
    sys.path.insert(0, SCRFD_VENDOR)

# Segformer プロジェクトルート (施策C で使用)
SEGFORMER_PROJ = os.environ.get("SEGFORMER_PROJ", "/home/pan/プロジェクト/02_GitHub/Segformer")
if SEGFORMER_PROJ not in sys.path:
    sys.path.insert(0, SEGFORMER_PROJ)

from config import Config
from mosaic_analyzer import analyze_roi
from scorer import (
    FrameVerdict,
    FaceResult,
    determine_face_verdict,
    determine_frame_verdict,
)

# テストセットのパス設定
TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = {
    "normal": TESTSET_ROOT / "normal",
    "gay":    TESTSET_ROOT / "gay",
    "hukusu": TESTSET_ROOT / "hukusu",
    "rez":    TESTSET_ROOT / "rez",
}


# ============================================================
# GT マスク生成
# ============================================================

def load_gt_mask(json_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """
    LabelMe JSON のポリゴンアノテーションから GT バイナリマスクを生成する。

    Returns:
        (H, W) uint8 マスク (1=顔領域, 0=背景)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") == "face":
            pts = np.array(shape["points"], dtype=np.float32)
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# 評価指標計算
# ============================================================

def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Coverage / Precision / IoU を計算する。

    Coverage (Recall) = TP / (TP + FN) = 検出した GT 顔ピクセル割合
    Precision         = TP / (TP + FP) = 検出ピクセル中の真の顔ピクセル割合
    IoU               = TP / (TP + FP + FN) = 標準的な二値 IoU

    Args:
        pred_mask: 予測マスク (0 or 1)
        gt_mask:   GT マスク   (0 or 1)

    Returns:
        {"coverage": float, "precision": float, "iou": float}
    """
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask > 0).astype(np.uint8)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & (~gt.astype(bool)))
    fn = np.sum((~pred.astype(bool)) & gt)

    coverage  = float(tp) / float(tp + fn)  if (tp + fn) > 0 else 0.0
    precision = float(tp) / float(tp + fp)  if (tp + fp) > 0 else 0.0
    iou       = float(tp) / float(tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"coverage": coverage, "precision": precision, "iou": iou}


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    フレームリストの metrics を平均する（空フレームは除く）。
    """
    if not metric_list:
        return {"coverage": 0.0, "precision": 0.0, "iou": 0.0}

    coverage  = float(np.mean([m["coverage"]  for m in metric_list]))
    precision = float(np.mean([m["precision"] for m in metric_list]))
    iou       = float(np.mean([m["iou"]       for m in metric_list]))
    return {"coverage": coverage, "precision": precision, "iou": iou}


# ============================================================
# 検出ロジック（フレーム 1 枚分）
# ============================================================

def extract_roi_from_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    padding: int = 10,
) -> np.ndarray:
    """bbox から ROI を切り出す（パディング付き）"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        roi = np.zeros((padding * 2, padding * 2, 3), dtype=np.uint8)
    return roi


def bbox_to_mask(
    bboxes: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    bboxes (N, 4) を塗りつぶしたバイナリマスク (H, W) に変換する。
    評価では「検出した bbox 領域」を予測マスクとして使う。
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for b in bboxes:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def segformer_bbox_to_mask(
    checker,
    frame: np.ndarray,
    bboxes: np.ndarray,
    img_h: int,
    img_w: int,
    padding: int = 10,
) -> np.ndarray:
    """
    施策C: SegFormer による精密マスク生成。
    各 bbox の ROI に SegFormer を適用し、検出した顔ピクセルを
    元画像座標に戻してフルマスクを構築する。
    """
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for b in bboxes:
        x1 = max(0, int(b[0]) - padding)
        y1 = max(0, int(b[1]) - padding)
        x2 = min(img_w, int(b[2]) + padding)
        y2 = min(img_h, int(b[3]) + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        face_mask_roi = checker.get_face_mask(roi)  # (roi_h, roi_w)
        # リサイズせず元サイズに戻す
        if face_mask_roi.shape != (y2 - y1, x2 - x1):
            face_mask_roi = cv2.resize(
                face_mask_roi, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
            )
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], face_mask_roi)
    return full_mask


def run_eval_frame(
    frame: np.ndarray,
    detector_25g,
    threshold_25g,
    cfg: Config,
    detector_34g=None,
    threshold_34g=None,
    segformer_checker=None,
) -> Tuple[np.ndarray, List[float]]:
    """
    1 フレームの顔検出 → verdict → 予測マスクを返す。

    Returns:
        pred_mask : (H, W) uint8  予測マスク (RED/YELLOW verdict の顔領域)
        confidences: 各検出 bbox の confidence スコアリスト
    """
    from scrfd.pub import SCRFD
    from PIL import Image

    h, w = frame.shape[:2]

    # 顔検出（施策D: ハイブリッド or 通常）
    from pipeline import detect_faces_scrfd, detect_faces_hybrid

    if cfg.use_scrfd34g_hybrid and detector_34g is not None:
        dets = detect_faces_hybrid(
            detector_25g, threshold_25g,
            detector_34g, threshold_34g,
            frame, cfg,
        )
    else:
        dets = detect_faces_scrfd(detector_25g, threshold_25g, frame, cfg)

    if len(dets) == 0:
        return np.zeros((h, w), dtype=np.uint8), []

    # 各 bbox の verdict を判定し、YELLOW/RED のみ予測マスクに含める
    flagged_bboxes = []
    confidences = []

    for det in dets:
        bbox = det[:4]
        score = float(det[4])

        roi = extract_roi_from_bbox(frame, bbox, cfg.roi_padding)
        analysis = analyze_roi(roi, cfg.roi_normalize_size)

        verdict = determine_face_verdict(
            laplacian_var=analysis["laplacian_var"],
            block_size=analysis["block_size"],
            periodicity=analysis["periodicity"],
            cfg=cfg,
            confidence=score,  # 施策B
        )

        # 施策C: SegFormer 二次確認（GREEN以外のみ対象）
        if segformer_checker is not None and verdict != FrameVerdict.GREEN:
            face_ratio = segformer_checker.face_pixel_ratio(roi)
            if face_ratio < cfg.segformer_face_ratio_min:
                verdict = FrameVerdict.GREEN

        if verdict != FrameVerdict.GREEN:
            flagged_bboxes.append(bbox)
            confidences.append(score)

    if not flagged_bboxes:
        return np.zeros((h, w), dtype=np.uint8), confidences

    # 予測マスク生成
    if segformer_checker is not None:
        # 施策C: SegFormer 精密マスク（評価精度向上のため bbox より小さいマスクを生成）
        pred_mask = segformer_bbox_to_mask(
            segformer_checker, frame,
            np.array(flagged_bboxes), h, w, cfg.roi_padding
        )
    else:
        # 通常: bbox を塗りつぶしたマスク
        pred_mask = bbox_to_mask(np.array(flagged_bboxes), h, w)

    return pred_mask, confidences


# ============================================================
# 施策Aのキャリブレーション解析
# ============================================================

def analyze_laplacian_calibration(
    testset_dir: Path,
    detector_25g,
    threshold_25g,
    cfg: Config,
    sample_limit: int = 200,
) -> Dict:
    """
    施策A: 実データ上で Laplacian 分布を計測し閾値キャリブレーション情報を返す。

    - 顔 ROI（no mosaic）の Laplacian 分布を計測
    - 合成モザイク（8px / 16px / 24px）の Laplacian 分布と比較
    """
    from pipeline import detect_faces_scrfd

    frame_files = sorted(testset_dir.glob("*.jpg"))[:sample_limit]
    laplacian_no_mosaic = []
    laplacian_mosaic_8  = []
    laplacian_mosaic_16 = []
    laplacian_mosaic_24 = []

    for img_path in tqdm(frame_files, desc=f"施策A calibration [{testset_dir.name}]"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        dets = detect_faces_scrfd(detector_25g, threshold_25g, frame, cfg)

        for det in dets:
            bbox = det[:4]
            roi = extract_roi_from_bbox(frame, bbox, cfg.roi_padding)
            if roi.size == 0:
                continue

            # 元画像（モザイクなし）の Laplacian
            a = analyze_roi(roi, cfg.roi_normalize_size)
            laplacian_no_mosaic.append(a["laplacian_var"])

            # 合成モザイクを適用した ROI の Laplacian
            for block_sz, store in [
                (8,  laplacian_mosaic_8),
                (16, laplacian_mosaic_16),
                (24, laplacian_mosaic_24),
            ]:
                roi_m = _apply_mosaic(roi, block_sz)
                a_m = analyze_roi(roi_m, cfg.roi_normalize_size)
                store.append(a_m["laplacian_var"])

    def stats(arr):
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "n": len(a),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p5":  float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        }

    return {
        "no_mosaic":    stats(laplacian_no_mosaic),
        "mosaic_8px":   stats(laplacian_mosaic_8),
        "mosaic_16px":  stats(laplacian_mosaic_16),
        "mosaic_24px":  stats(laplacian_mosaic_24),
    }


def _apply_mosaic(img: np.ndarray, block_size: int) -> np.ndarray:
    """指定ブロックサイズのモザイクを img に適用して返す"""
    h, w = img.shape[:2]
    if block_size <= 0 or h == 0 or w == 0:
        return img
    small = cv2.resize(
        img, (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ============================================================
# メイン評価ループ
# ============================================================

def evaluate_config(
    config_name: str,
    cfg: Config,
    detector_25g,
    threshold_25g,
    detector_34g=None,
    threshold_34g=None,
    segformer_checker=None,
) -> Dict:
    """全テストセットで評価を実行し、結果辞書を返す"""
    results = {}

    for subset_name, subset_dir in TESTSETS.items():
        if not subset_dir.exists():
            print(f"  [SKIP] {subset_dir} が存在しません")
            continue

        frame_files = sorted(subset_dir.glob("*.jpg"))
        subset_metrics = []
        n_gt_frames = 0

        for img_path in tqdm(frame_files, desc=f"  [{subset_name}] {config_name}"):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]

            # GT マスク生成
            gt_mask = load_gt_mask(json_path, img_h, img_w)
            if gt_mask.sum() == 0:
                # 顔アノテーションなし → スキップ
                continue
            n_gt_frames += 1

            # 予測マスク生成
            pred_mask, _ = run_eval_frame(
                frame, detector_25g, threshold_25g, cfg,
                detector_34g, threshold_34g, segformer_checker,
            )

            # 指標計算
            m = compute_metrics(pred_mask, gt_mask)
            subset_metrics.append(m)

        agg = aggregate_metrics(subset_metrics)
        agg["n_frames"] = n_gt_frames
        results[subset_name] = agg
        print(
            f"    {subset_name}: n={n_gt_frames}"
            f"  Coverage={agg['coverage']:.4f}"
            f"  Precision={agg['precision']:.4f}"
            f"  IoU={agg['iou']:.4f}"
        )

    # 加重平均（フレーム数ベース）
    total_n = sum(v["n_frames"] for v in results.values())
    if total_n > 0:
        w_coverage  = sum(v["coverage"]  * v["n_frames"] for v in results.values()) / total_n
        w_precision = sum(v["precision"] * v["n_frames"] for v in results.values()) / total_n
        w_iou       = sum(v["iou"]       * v["n_frames"] for v in results.values()) / total_n
        results["weighted_avg"] = {
            "coverage": w_coverage,
            "precision": w_precision,
            "iou": w_iou,
            "n_frames": total_n,
        }

    return results


# ============================================================
# エントリポイント
# ============================================================

def build_config(config_name: str) -> Config:
    """設定名から Config を構築する"""
    cfg = Config()
    # 評価用に通常の閾値を使用
    cfg.scrfd_conf_th = 0.3

    if config_name == "施策B":
        cfg.use_confidence_boost = True
    elif config_name == "施策C":
        cfg.use_segformer_confirmation = True
    elif config_name == "施策D":
        cfg.use_scrfd34g_hybrid = True
    elif config_name == "all":
        cfg.use_confidence_boost = True
        cfg.use_segformer_confirmation = True
        cfg.use_scrfd34g_hybrid = True
    elif config_name == "baseline":
        # baseline は旧デフォルト（SCRFD-2.5g のみ）
        cfg.use_scrfd34g_hybrid = False
        cfg.use_confidence_boost = False
    elif config_name == "production":
        # 本番想定（conf_th=0.50 + hybrid）
        cfg.scrfd_conf_th = 0.50
        cfg.use_scrfd34g_hybrid = True

    return cfg


def main():
    parser = argparse.ArgumentParser(description="testset_2603 評価スクリプト")
    parser.add_argument(
        "--config", nargs="+",
        default=["baseline", "施策B", "施策C", "施策D", "all"],
        help="評価する設定名（baseline / 施策B / 施策C / 施策D / all）"
    )
    parser.add_argument(
        "--out", default=None,
        help="結果 JSON 出力パス（省略時はコンソール出力のみ）"
    )
    parser.add_argument(
        "--skip-calib", action="store_true",
        help="施策A キャリブレーション解析をスキップ"
    )
    args = parser.parse_args()

    all_results = {}
    t0 = time.time()

    # --- 施策A: Laplacian キャリブレーション解析 ---
    if not args.skip_calib:
        print("\n=== 施策A: Laplacian キャリブレーション解析 ===")
        cfg_base = Config()
        from scrfd.pub import SCRFD
        from scrfd.schemas import Threshold

        det25 = SCRFD.from_path(cfg_base.scrfd_model_path, providers=list(cfg_base.scrfd_providers))
        th25  = Threshold(probability=cfg_base.scrfd_conf_th, nms=cfg_base.scrfd_nms_th)

        calib_results = {}
        for sname, sdir in TESTSETS.items():
            if not sdir.exists():
                continue
            calib_results[sname] = analyze_laplacian_calibration(sdir, det25, th25, cfg_base)

        all_results["calibration_laplacian"] = calib_results

        # キャリブレーション結果の表示
        print("\n--- Laplacian 統計サマリー (normal subset) ---")
        nc = calib_results.get("normal", {})
        for key, stats in nc.items():
            if stats:
                print(
                    f"  {key:15s}: mean={stats.get('mean', 0):.1f}"
                    f"  median={stats.get('median', 0):.1f}"
                    f"  p5={stats.get('p5', 0):.1f}"
                    f"  p95={stats.get('p95', 0):.1f}"
                )

    # --- 各設定の評価 ---
    print("\n=== 評価設定ごとの Coverage / Precision / IoU ===")
    all_eval_results = {}

    for config_name in args.config:
        print(f"\n--- 設定: {config_name} ---")
        cfg = build_config(config_name)

        from scrfd.pub import SCRFD
        from scrfd.schemas import Threshold

        # 25g 検出器（全設定共通）
        det25 = SCRFD.from_path(cfg.scrfd_model_path, providers=list(cfg.scrfd_providers))
        th25  = Threshold(probability=cfg.scrfd_conf_th, nms=cfg.scrfd_nms_th)

        # 施策D: 34g 検出器
        det34, th34 = None, None
        if cfg.use_scrfd34g_hybrid:
            det34 = SCRFD.from_path(cfg.scrfd_34g_model_path, providers=list(cfg.scrfd_providers))
            th34  = Threshold(probability=cfg.scrfd_conf_th, nms=cfg.scrfd_nms_th)

        # 施策C: SegFormer チェッカー
        segf = None
        if cfg.use_segformer_confirmation:
            from segformer_checker import SegFormerChecker
            segf = SegFormerChecker(
                model_path=cfg.segformer_model_path,
                face_classes=cfg.segformer_face_classes,
                device="cuda",
            )

        result = evaluate_config(
            config_name, cfg, det25, th25, det34, th34, segf
        )
        all_eval_results[config_name] = result

        if "weighted_avg" in result:
            wa = result["weighted_avg"]
            print(
                f"  【{config_name} 加重平均】"
                f" Coverage={wa['coverage']:.4f}"
                f" Precision={wa['precision']:.4f}"
                f" IoU={wa['iou']:.4f}"
                f" (n={wa['n_frames']})"
            )

    all_results["evaluation"] = all_eval_results
    elapsed = time.time() - t0
    all_results["elapsed_sec"] = round(elapsed, 1)

    # --- 比較サマリー表示 ---
    print("\n=== サマリー比較表 ===")
    print(f"{'設定':12s} {'Coverage':>10s} {'Precision':>10s} {'IoU':>10s}")
    print("-" * 46)
    for cname, cres in all_eval_results.items():
        wa = cres.get("weighted_avg", {})
        if wa:
            print(
                f"{cname:12s}"
                f" {wa['coverage']:10.4f}"
                f" {wa['precision']:10.4f}"
                f" {wa['iou']:10.4f}"
            )

    # --- JSON 出力 ---
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n結果を保存: {out_path}")

    print(f"\n総処理時間: {elapsed:.1f}秒")
    return all_results


if __name__ == "__main__":
    main()
