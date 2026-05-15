"""性器検出 FP のアスペクト比分析

testset_2603 でGroundingDinoの検出BBoxとGTを比較し、
TP/FP それぞれのアスペクト比分布を分析する。

目的: Aspect Ratio フィルタによるFP削減の可能性を定量的に検証する。
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# --- パス設定 ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
GSAM2_DIR = os.path.join(REPO_DIR, "genital-reference", "LV_Grounded-SAM-2")

sys.path.insert(0, SRC_DIR)
_gdino_pkg = os.path.join(GSAM2_DIR, "grounding_dino")
if os.path.isdir(_gdino_pkg) and _gdino_pkg not in sys.path:
    sys.path.insert(0, _gdino_pkg)
if GSAM2_DIR not in sys.path:
    sys.path.insert(0, GSAM2_DIR)

TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = {
    "normal": TESTSET_ROOT / "normal",
    "gay":    TESTSET_ROOT / "gay",
    "hukusu": TESTSET_ROOT / "hukusu",
    "rez":    TESTSET_ROOT / "rez",
}

GENITAL_LABELS = {"vagina_wide", "penis", "anus_insertion"}
GENITAL_TEXT_PROMPT = "vagina . penis ."
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.15
MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.30


def load_gt_bboxes(json_path: Path, img_h: int, img_w: int) -> List[Tuple[int,int,int,int]]:
    """GT ポリゴンから外接矩形を取得"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bboxes = []
    for shape in data.get("shapes", []):
        if shape.get("label") in GENITAL_LABELS:
            pts = np.array(shape["points"], dtype=np.float32)
            x1 = max(0, int(pts[:, 0].min()))
            y1 = max(0, int(pts[:, 1].min()))
            x2 = min(img_w, int(pts[:, 0].max()))
            y2 = min(img_h, int(pts[:, 1].max()))
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def compute_iou_bbox(box1, box2) -> float:
    """2つのbbox (x1,y1,x2,y2) のIoUを計算"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def aspect_ratio(bbox) -> float:
    """max(w/h, h/w) を返す（常に >= 1.0）"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return 1.0
    return max(w / h, h / w)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-checkpoint", default=os.path.join(
        GSAM2_DIR, "gdino_checkpoints", "dino_local_ft_ep4_best.pth"))
    parser.add_argument("--dino-config", default=os.path.join(GSAM2_DIR, "cfg_odvg.py"))
    parser.add_argument("--iou-threshold", type=float, default=0.1,
                       help="IoU threshold for TP/FP classification")
    args = parser.parse_args()

    print("=" * 60)
    print("性器検出 TP/FP アスペクト比分析")
    print("=" * 60)
    print(f"  checkpoint: {os.path.basename(args.dino_checkpoint)}")
    print(f"  IoU threshold: {args.iou_threshold}")

    # モデルロード
    from groundingdino.util.inference import load_model, predict
    import torch
    import torchvision.transforms as T
    from torchvision.ops import box_convert
    from PIL import Image as PILImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.dino_config, args.dino_checkpoint, device=device)
    model = model.to(device)
    print(f"  device: {device}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tp_ars = []  # TP detections のアスペクト比
    fp_ars = []  # FP detections のアスペクト比
    tp_scores = []
    fp_scores = []
    total_gt = 0
    total_det = 0

    for subset_name, subset_dir in TESTSETS.items():
        if not subset_dir.exists():
            continue

        frame_files = sorted(subset_dir.glob("*.jpg"))
        print(f"\n  [{subset_name}] {len(frame_files)} files...", end="", flush=True)

        for img_path in frame_files:
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            h, w = frame.shape[:2]
            gt_bboxes = load_gt_bboxes(json_path, h, w)
            if not gt_bboxes:
                continue

            total_gt += len(gt_bboxes)

            # 検出
            rgb_pil = PILImage.fromarray(frame[:, :, ::-1])
            image_tensor = transform(rgb_pil)

            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=model, image=image_tensor,
                    caption=GENITAL_TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=device,
                )

            if len(boxes) == 0:
                continue

            boxes_xyxy = box_convert(
                boxes=boxes * torch.tensor([w, h, w, h], dtype=torch.float32),
                in_fmt="cxcywh", out_fmt="xyxy",
            ).numpy()

            for box, score in zip(boxes_xyxy, logits.numpy()):
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                box_area = (x2 - x1) * (y2 - y1)
                if box_area < w * h * MIN_BOX_AREA_RATIO:
                    continue
                if box_area > w * h * MAX_BOX_AREA_RATIO:
                    continue

                total_det += 1
                det_bbox = (x1, y1, x2, y2)
                ar = aspect_ratio(det_bbox)

                # TP/FP分類: GTとのIoUが閾値以上ならTP
                max_iou = max(compute_iou_bbox(det_bbox, gt) for gt in gt_bboxes)

                if max_iou >= args.iou_threshold:
                    tp_ars.append(ar)
                    tp_scores.append(float(score))
                else:
                    fp_ars.append(ar)
                    fp_scores.append(float(score))

        print(f" done", flush=True)

    # --- 結果出力 ---
    print("\n" + "=" * 60)
    print("結果サマリ")
    print("=" * 60)
    print(f"  総GT数: {total_gt}")
    print(f"  総検出数: {total_det}")
    print(f"  TP検出数: {len(tp_ars)}")
    print(f"  FP検出数: {len(fp_ars)}")
    print(f"  FP率: {len(fp_ars)/max(total_det,1)*100:.1f}%")

    if tp_ars:
        tp_arr = np.array(tp_ars)
        print(f"\n  TP アスペクト比 (max(w/h, h/w)):")
        print(f"    mean={tp_arr.mean():.2f}, median={np.median(tp_arr):.2f}")
        print(f"    p25={np.percentile(tp_arr,25):.2f}, p75={np.percentile(tp_arr,75):.2f}, p95={np.percentile(tp_arr,95):.2f}, max={tp_arr.max():.2f}")
        print(f"    AR>3: {(tp_arr>3).sum()} ({(tp_arr>3).sum()/len(tp_arr)*100:.1f}%)")
        print(f"    AR>4: {(tp_arr>4).sum()} ({(tp_arr>4).sum()/len(tp_arr)*100:.1f}%)")
        print(f"    AR>5: {(tp_arr>5).sum()} ({(tp_arr>5).sum()/len(tp_arr)*100:.1f}%)")

    if fp_ars:
        fp_arr = np.array(fp_ars)
        print(f"\n  FP アスペクト比 (max(w/h, h/w)):")
        print(f"    mean={fp_arr.mean():.2f}, median={np.median(fp_arr):.2f}")
        print(f"    p25={np.percentile(fp_arr,25):.2f}, p75={np.percentile(fp_arr,75):.2f}, p95={np.percentile(fp_arr,95):.2f}, max={fp_arr.max():.2f}")
        print(f"    AR>3: {(fp_arr>3).sum()} ({(fp_arr>3).sum()/len(fp_arr)*100:.1f}%)")
        print(f"    AR>4: {(fp_arr>4).sum()} ({(fp_arr>4).sum()/len(fp_arr)*100:.1f}%)")
        print(f"    AR>5: {(fp_arr>5).sum()} ({(fp_arr>5).sum()/len(fp_arr)*100:.1f}%)")

    if fp_scores:
        fp_score_arr = np.array(fp_scores)
        print(f"\n  FP スコア分布:")
        print(f"    mean={fp_score_arr.mean():.3f}, median={np.median(fp_score_arr):.3f}")
        print(f"    p25={np.percentile(fp_score_arr,25):.3f}, p75={np.percentile(fp_score_arr,75):.3f}")

    if tp_scores:
        tp_score_arr = np.array(tp_scores)
        print(f"\n  TP スコア分布:")
        print(f"    mean={tp_score_arr.mean():.3f}, median={np.median(tp_score_arr):.3f}")
        print(f"    p25={np.percentile(tp_score_arr,25):.3f}, p75={np.percentile(tp_score_arr,75):.3f}")

    # AR閾値候補ごとの影響シミュレーション
    print("\n" + "-" * 60)
    print("AR フィルタ閾値候補の影響シミュレーション")
    print("-" * 60)
    print(f"{'AR閾値':<8} {'FP削減数':<10} {'FP削減率':<10} {'TP損失数':<10} {'TP損失率':<10}")
    for ar_th in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]:
        fp_removed = sum(1 for a in fp_ars if a > ar_th) if fp_ars else 0
        tp_lost = sum(1 for a in tp_ars if a > ar_th) if tp_ars else 0
        fp_pct = fp_removed / max(len(fp_ars), 1) * 100
        tp_pct = tp_lost / max(len(tp_ars), 1) * 100
        print(f"  >{ar_th:<5.1f} {fp_removed:<10} {fp_pct:<9.1f}% {tp_lost:<10} {tp_pct:<9.1f}%")


if __name__ == "__main__":
    main()
