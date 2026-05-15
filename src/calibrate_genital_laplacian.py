"""性器パイプライン — Laplacian 閾値キャリブレーション & 偽陽性分析

testset_2603 の GT アノテーション付き画像に対して Fine-tuned GroundingDino を実行し、
以下を分析する:

1. **Laplacian キャリブレーション**: 性器 GT 領域の Laplacian 分散値分布を測定
   - GT bbox の ROI を切り出して Laplacian/ブロックサイズ/DCT 周期性を計測
   - モザイク適用済み動画用の閾値推奨値を算出

2. **偽陽性分析**: GroundingDino の検出結果が GT と重なるかを確認
   - GT 性器領域と重なる検出 = True Positive（TP）
   - GT 性器領域と重ならない検出 = False Positive（FP: 腕/脚/腹部等の誤検出）
   - FP の body part を分析

Usage:
    cd /home/pan/プロジェクト/16.モザイク検知
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/pan/miniforge3/envs/ml/bin/python src/calibrate_genital_laplacian.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# パス設定
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(REPO_DIR, "models")
VENDOR_DIR = os.path.join(REPO_DIR, "vendor")

sys.path.insert(0, SRC_DIR)
_gdino_lib = os.path.join(VENDOR_DIR, "grounding_dino")
if os.path.isdir(_gdino_lib) and _gdino_lib not in sys.path:
    sys.path.insert(0, _gdino_lib)
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from mosaic_analyzer import analyze_roi

# --- 定数 ---
TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
SUBSETS = ["normal", "gay", "hukusu", "rez"]
GENITAL_LABELS = {"vagina_wide", "penis", "anus_insertion"}
FACE_LABEL = "face"

DEFAULT_DINO_CHECKPOINT = os.path.join(
    GSAM2_DIR, "gdino_checkpoints", "dino_local_ft_ep4_best.pth"
)
DEFAULT_DINO_CONFIG = os.path.join(MODELS_DIR, "gdino", "cfg_odvg.py")
GENITAL_TEXT_PROMPT = "vagina . penis ."
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.15
ROI_PADDING = 10
ROI_NORMALIZE_SIZE = 64


def load_gt_bboxes(json_path: str, labels: set) -> List[Dict]:
    """LabelMe JSON からポリゴン → bbox を取得"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bboxes = []
    for shape in data.get("shapes", []):
        if shape.get("label") in labels:
            pts = np.array(shape["points"], dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            bboxes.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "label": shape["label"],
            })
    return bboxes


def compute_iou(box_a, box_b) -> float:
    """2つの bbox の IoU を計算"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_roi(frame, bbox, padding=10):
    """ROI 切り出し"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((padding * 2, padding * 2, 3), dtype=np.uint8)
    return roi


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="各サブセットの最大フレーム数（デバッグ用）")
    args = parser.parse_args()

    print("=" * 70)
    print("性器パイプライン — Laplacian キャリブレーション & 偽陽性分析")
    print("=" * 70)
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  box_threshold: {args.box_threshold}")

    # --- モデルロード ---
    print("\n[1] モデルロード中...")
    t0 = time.time()
    from groundingdino.util.inference import load_model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.config, args.checkpoint, device=device)
    model = model.to(device)
    print(f"    ロード時間: {time.time()-t0:.1f}s, device={device}")

    # --- 結果格納 ---
    # Laplacian キャリブレーション: GT 性器 bbox の ROI 解析
    gt_laplacian_stats = []  # {subset, label, laplacian_var, block_size, periodicity}

    # 偽陽性分析: 検出と GT の対応
    tp_count = 0
    fp_count = 0
    fn_count = 0
    fp_details = []  # FP の詳細 {subset, file, bbox, score, iou_max}
    tp_laplacian_stats = []  # TP 検出の Laplacian

    # 顔 GT bbox と重なる FP
    fp_face_overlap = 0
    fp_no_gt_overlap = 0

    subset_stats = {}

    print("\n[2] 分析中...")
    for subset_name in SUBSETS:
        subset_dir = TESTSET_ROOT / subset_name
        if not subset_dir.exists():
            print(f"  [SKIP] {subset_name}")
            continue

        img_files = sorted(subset_dir.glob("*.jpg"))
        if args.max_frames:
            img_files = img_files[:args.max_frames]

        s_tp, s_fp, s_fn = 0, 0, 0
        s_fp_face, s_fp_ngt = 0, 0

        print(f"\n  [{subset_name}] ({len(img_files)} frames)", end="", flush=True)
        t1 = time.time()

        for img_path in img_files:
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # GT bbox 取得
            gt_genital = load_gt_bboxes(str(json_path), GENITAL_LABELS)
            gt_face = load_gt_bboxes(str(json_path), {FACE_LABEL})

            # --- Part 1: GT 性器 bbox の Laplacian 計測 ---
            for gt in gt_genital:
                roi = extract_roi(frame, gt["bbox"], ROI_PADDING)
                analysis = analyze_roi(roi, ROI_NORMALIZE_SIZE)
                gt_laplacian_stats.append({
                    "subset": subset_name,
                    "label": gt["label"],
                    "laplacian_var": analysis["laplacian_var"],
                    "block_size": analysis["block_size"],
                    "periodicity": analysis["periodicity"],
                })

            # --- Part 2: GroundingDino 検出 → TP/FP 分析 ---
            from PIL import Image as PILImage
            import torchvision.transforms as T
            from torchvision.ops import box_convert
            from groundingdino.util.inference import predict

            rgb_pil = PILImage.fromarray(frame[:, :, ::-1])
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor = transform(rgb_pil)

            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=model, image=image_tensor,
                    caption=GENITAL_TEXT_PROMPT,
                    box_threshold=args.box_threshold,
                    text_threshold=TEXT_THRESHOLD,
                    device=device,
                )

            # 検出結果を xyxy に変換
            det_results = []
            if len(boxes) > 0:
                boxes_xyxy = box_convert(
                    boxes=boxes * torch.tensor([w, h, w, h], dtype=torch.float32),
                    in_fmt="cxcywh", out_fmt="xyxy",
                ).numpy()
                for box, score, label in zip(boxes_xyxy, logits.numpy(), phrases):
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    area = (x2 - x1) * (y2 - y1)
                    if area < w * h * 0.001:
                        continue
                    det_results.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": float(score),
                        "label": label.strip(),
                    })

            # TP/FP マッチング (IoU > 0.1 で GT にマッチ)
            gt_matched = set()
            for det in det_results:
                best_iou = 0
                best_gt_idx = -1
                for gi, gt in enumerate(gt_genital):
                    iou = compute_iou(det["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_iou > 0.1:
                    # True Positive
                    tp_count += 1
                    s_tp += 1
                    gt_matched.add(best_gt_idx)

                    # TP 検出の Laplacian 計測
                    roi = extract_roi(frame, det["bbox"], ROI_PADDING)
                    analysis = analyze_roi(roi, ROI_NORMALIZE_SIZE)
                    tp_laplacian_stats.append({
                        "subset": subset_name,
                        "score": det["score"],
                        "laplacian_var": analysis["laplacian_var"],
                        "block_size": analysis["block_size"],
                        "periodicity": analysis["periodicity"],
                    })
                else:
                    # False Positive
                    fp_count += 1
                    s_fp += 1

                    # 顔 GT と重なるかチェック
                    face_iou_max = 0
                    for gf in gt_face:
                        face_iou_max = max(face_iou_max, compute_iou(det["bbox"], gf["bbox"]))

                    if face_iou_max > 0.1:
                        fp_face_overlap += 1
                        s_fp_face += 1
                    else:
                        fp_no_gt_overlap += 1
                        s_fp_ngt += 1

                    fp_details.append({
                        "subset": subset_name,
                        "file": img_path.name,
                        "bbox": det["bbox"],
                        "score": det["score"],
                        "iou_max_genital": best_iou,
                        "iou_max_face": face_iou_max,
                    })

            # False Negative (GT にマッチしなかった)
            fn = len(gt_genital) - len(gt_matched)
            fn_count += fn
            s_fn += fn

        elapsed = time.time() - t1
        subset_stats[subset_name] = {
            "n_frames": len(img_files),
            "tp": s_tp, "fp": s_fp, "fn": s_fn,
            "fp_face": s_fp_face, "fp_no_gt": s_fp_ngt,
            "time": elapsed,
        }
        print(f"  TP={s_tp}, FP={s_fp}(face={s_fp_face},other={s_fp_ngt}), FN={s_fn} ({elapsed:.0f}s)")

    # ============================================================
    # 結果レポート
    # ============================================================
    print("\n" + "=" * 70)
    print("結果レポート")
    print("=" * 70)

    # --- Laplacian キャリブレーション ---
    print("\n=== Part 1: GT 性器領域の Laplacian 分布 ===")
    print("（testset_2603 はモザイク未適用の原画フレーム）\n")

    if gt_laplacian_stats:
        all_lap = [s["laplacian_var"] for s in gt_laplacian_stats]
        all_bs = [s["block_size"] for s in gt_laplacian_stats if s["block_size"] is not None]
        all_per = [s["periodicity"] for s in gt_laplacian_stats]

        print(f"  全サンプル数: {len(gt_laplacian_stats)}")
        print(f"  Laplacian 分散値:")
        print(f"    mean={np.mean(all_lap):.1f}, median={np.median(all_lap):.1f}")
        print(f"    p5={np.percentile(all_lap, 5):.1f}, p25={np.percentile(all_lap, 25):.1f}")
        print(f"    p75={np.percentile(all_lap, 75):.1f}, p95={np.percentile(all_lap, 95):.1f}")
        print(f"    min={np.min(all_lap):.1f}, max={np.max(all_lap):.1f}")

        if all_bs:
            print(f"  ブロックサイズ:")
            print(f"    mean={np.mean(all_bs):.1f}, median={np.median(all_bs):.1f}")
            print(f"    None率: {1 - len(all_bs)/len(gt_laplacian_stats):.1%}")

        print(f"  DCT周期性:")
        print(f"    mean={np.mean(all_per):.2f}, median={np.median(all_per):.2f}")

        # ラベル別
        print("\n  --- ラベル別 Laplacian ---")
        for label in sorted(set(s["label"] for s in gt_laplacian_stats)):
            laps = [s["laplacian_var"] for s in gt_laplacian_stats if s["label"] == label]
            print(f"    {label}: n={len(laps)}, mean={np.mean(laps):.1f}, "
                  f"median={np.median(laps):.1f}, "
                  f"p5={np.percentile(laps, 5):.1f}, p95={np.percentile(laps, 95):.1f}")

        # サブセット別
        print("\n  --- サブセット別 Laplacian ---")
        for subset in SUBSETS:
            laps = [s["laplacian_var"] for s in gt_laplacian_stats if s["subset"] == subset]
            if laps:
                print(f"    {subset}: n={len(laps)}, mean={np.mean(laps):.1f}, "
                      f"median={np.median(laps):.1f}")

    # --- TP 検出の Laplacian ---
    print("\n=== Part 1b: TP 検出（GT と重なった検出）の Laplacian 分布 ===\n")
    if tp_laplacian_stats:
        tp_laps = [s["laplacian_var"] for s in tp_laplacian_stats]
        tp_scores = [s["score"] for s in tp_laplacian_stats]
        print(f"  TP 検出数: {len(tp_laplacian_stats)}")
        print(f"  Laplacian 分散値:")
        print(f"    mean={np.mean(tp_laps):.1f}, median={np.median(tp_laps):.1f}")
        print(f"    p5={np.percentile(tp_laps, 5):.1f}, p95={np.percentile(tp_laps, 95):.1f}")
        print(f"  GDino スコア:")
        print(f"    mean={np.mean(tp_scores):.3f}, median={np.median(tp_scores):.3f}")
        print(f"    p5={np.percentile(tp_scores, 5):.3f}, p95={np.percentile(tp_scores, 95):.3f}")

    # --- 偽陽性分析 ---
    print("\n=== Part 2: 偽陽性 (FP) 分析 ===\n")
    total_det = tp_count + fp_count
    print(f"  全検出数: {total_det}")
    print(f"    TP (GT 性器と重なる): {tp_count} ({tp_count/max(total_det,1):.1%})")
    print(f"    FP (GT と重ならない): {fp_count} ({fp_count/max(total_det,1):.1%})")
    print(f"      FP うち 顔 GT と重なる: {fp_face_overlap} (顔を誤検出)")
    print(f"      FP うち どの GT とも重ならない: {fp_no_gt_overlap} (背景/体)")
    print(f"    FN (未検出 GT): {fn_count}")

    if total_det > 0:
        precision = tp_count / total_det
        recall = tp_count / max(tp_count + fn_count, 1)
        print(f"\n  Precision (bbox): {precision:.3f}")
        print(f"  Recall (bbox): {recall:.3f}")

    # FP のスコア分布
    if fp_details:
        fp_scores = [d["score"] for d in fp_details]
        print(f"\n  FP スコア分布:")
        print(f"    mean={np.mean(fp_scores):.3f}, median={np.median(fp_scores):.3f}")
        print(f"    p5={np.percentile(fp_scores, 5):.3f}, p95={np.percentile(fp_scores, 95):.3f}")

        print(f"\n  --- FP サブセット別 ---")
        for subset in SUBSETS:
            subs = [d for d in fp_details if d["subset"] == subset]
            if subs:
                n_face = sum(1 for d in subs if d["iou_max_face"] > 0.1)
                n_other = len(subs) - n_face
                print(f"    {subset}: FP={len(subs)} (顔={n_face}, その他={n_other})")

    # サブセット別サマリ
    print("\n=== サブセット別サマリ ===\n")
    print(f"  {'subset':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'FP_face':>8} {'FP_other':>8} {'Prec':>6} {'Recall':>6}")
    print(f"  {'-'*64}")
    for subset in SUBSETS:
        if subset in subset_stats:
            s = subset_stats[subset]
            prec = s['tp'] / max(s['tp'] + s['fp'], 1)
            rec = s['tp'] / max(s['tp'] + s['fn'], 1)
            print(f"  {subset:<10} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5} "
                  f"{s['fp_face']:>8} {s['fp_no_gt']:>8} {prec:>6.3f} {rec:>6.3f}")

    # 推奨閾値
    print("\n=== 推奨閾値 ===\n")
    if gt_laplacian_stats:
        all_lap = [s["laplacian_var"] for s in gt_laplacian_stats]
        print("  ※ testset_2603 はモザイク未適用の原画。")
        print("  ※ 以下は「原画（=モザイクなし）の性器 ROI の Laplacian 分布」であり、")
        print("    モザイク適用済みの性器 ROI は p5 よりもさらに低い値になる。")
        print(f"\n  原画 GT 性器 ROI の p5 = {np.percentile(all_lap, 5):.1f}")
        print(f"  原画 GT 性器 ROI の p25 = {np.percentile(all_lap, 25):.1f}")
        print(f"\n  → laplacian_threshold_high（RED判定）の推奨値:")
        print(f"    原画の p5 ≈ {np.percentile(all_lap, 5):.0f} を採用")
        print(f"    （原画でこの値未満は「薄い性器画像」→ モザイクなし時でも Laplacian が低い）")
        print(f"  → laplacian_threshold_low（GREEN判定）は実モザイク動画で要検証")

    # JSON 出力
    output_path = os.path.join(REPO_DIR, "genital-reference", "LV_Open-GroundingDino",
                               "local_train", "calibration_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "gt_laplacian_summary": {
                "n": len(gt_laplacian_stats),
                "mean": float(np.mean(all_lap)) if gt_laplacian_stats else 0,
                "median": float(np.median(all_lap)) if gt_laplacian_stats else 0,
                "p5": float(np.percentile(all_lap, 5)) if gt_laplacian_stats else 0,
                "p25": float(np.percentile(all_lap, 25)) if gt_laplacian_stats else 0,
                "p75": float(np.percentile(all_lap, 75)) if gt_laplacian_stats else 0,
                "p95": float(np.percentile(all_lap, 95)) if gt_laplacian_stats else 0,
            },
            "detection_summary": {
                "total": total_det,
                "tp": tp_count, "fp": fp_count, "fn": fn_count,
                "fp_face_overlap": fp_face_overlap,
                "fp_no_gt_overlap": fp_no_gt_overlap,
                "precision": tp_count / max(total_det, 1),
                "recall": tp_count / max(tp_count + fn_count, 1),
            },
            "subset_stats": subset_stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n結果 JSON: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
