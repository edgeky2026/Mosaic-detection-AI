#!/usr/bin/env python3
"""
SCRFD FT チェックポイント一括評価スクリプト

testset_2603 で各エポックの ONNX モデルを評価し、
Coverage / Precision / IoU / F1 を比較する。

GT は testset_2603/{subset}/ 直下の .json (LabelMe polygon 形式)。
画像は同ディレクトリの .jpg。
"""

import argparse
import json
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
VENDOR_DIR = os.path.join(REPO_DIR, 'vendor')
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# --- パス設定 ---
SCRFD_VENDOR = VENDOR_DIR
if SCRFD_VENDOR not in sys.path:
    sys.path.insert(0, SCRFD_VENDOR)

from scrfd.base import SCRFDBase
from scrfd.schemas import Threshold

TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = ["normal", "gay", "hukusu", "rez"]


def load_gt_mask(json_path: Path, h: int, w: int) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") == "face":
            pts = np.array(shape["points"], dtype=np.float32).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def bboxes_to_mask(bboxes: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = pred > 0
    g = gt > 0
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    cov = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"coverage": cov, "precision": prec, "iou": iou}


def create_detector(model_path: str):
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    return SCRFDBase.from_session(session)


def detect(model: SCRFDBase, bgr: np.ndarray, conf_th: float) -> np.ndarray:
    rgb = Image.fromarray(bgr[:, :, ::-1])
    th = Threshold(probability=conf_th, nms=0.5)
    result = model.detect(rgb, th)
    return result.bboxes  # Nx5 [x1,y1,x2,y2,score]


def evaluate_model(model_path: str, conf_th: float = 0.15) -> dict:
    model = create_detector(model_path)
    all_metrics = []

    for subset in TESTSETS:
        ts_dir = TESTSET_ROOT / subset
        frame_files = sorted(ts_dir.glob("*.jpg"))

        for img_path in frame_files:
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            gt = load_gt_mask(json_path, h, w)
            if gt.sum() == 0:
                continue

            bboxes = detect(model, bgr, conf_th)
            pred = bboxes_to_mask(bboxes, h, w) if len(bboxes) > 0 else np.zeros((h, w), dtype=np.uint8)
            m = compute_metrics(pred, gt)
            all_metrics.append(m)

    if not all_metrics:
        return {"coverage": 0, "precision": 0, "iou": 0, "f1": 0, "n": 0}

    cov = np.mean([m["coverage"] for m in all_metrics])
    prec = np.mean([m["precision"] for m in all_metrics])
    iou = np.mean([m["iou"] for m in all_metrics])
    f1 = 2 * cov * prec / (cov + prec) if (cov + prec) > 0 else 0

    return {
        "coverage": round(float(cov), 4),
        "precision": round(float(prec), 4),
        "iou": round(float(iou), 4),
        "f1": round(float(f1), 4),
        "n": len(all_metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--pattern", default="*.onnx")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    onnx_files = sorted(onnx_dir.glob(args.pattern))

    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir} with pattern {args.pattern}")
        return

    results = {}
    for onnx_path in onnx_files:
        name = onnx_path.stem
        print(f"\n=== Evaluating: {name} (conf={args.conf}) ===")
        t0 = time.time()
        r = evaluate_model(str(onnx_path), conf_th=args.conf)
        elapsed = time.time() - t0
        r["elapsed_sec"] = round(elapsed, 1)
        results[name] = r
        print(f"  Coverage={r['coverage']:.4f}  Precision={r['precision']:.4f}  "
              f"IoU={r['iou']:.4f}  F1={r['f1']:.4f}  n={r['n']}  "
              f"({elapsed:.1f}s)")

    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Coverage':>9} {'Precision':>10} {'IoU':>7} {'F1':>7}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30} {r['coverage']:>9.4f} {r['precision']:>10.4f} "
              f"{r['iou']:>7.4f} {r['f1']:>7.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
