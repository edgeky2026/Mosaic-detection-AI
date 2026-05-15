#!/usr/bin/env python3
"""UNION 検出（SCRFD-34g + RetinaFace R50）の実測評価

個別モデル（SCRFD-2.5g, SCRFD-34g, RetinaFace R50）に加えて、
SCRFD-34g + RetinaFace R50 の UNION（OR 結合）を同一条件で評価し、
Coverage, Precision, IoU, F1 を比較する。

testset_2603（normal/gay/hukusu/rez, 計 ~2,443 GT フレーム）を使用。

使用方法:
    python eval_union_compare.py                       # conf=0.15 で全実行
    python eval_union_compare.py --conf 0.15 0.30      # 複数 conf
    python eval_union_compare.py --max-frames 50       # クイックテスト
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- パス設定 ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

VENDOR_DIR = os.path.dirname(SRC_DIR) + "/vendor"
RETINAFACE_DIR = VENDOR_DIR + "/Pytorch_Retinaface"
if RETINAFACE_DIR not in sys.path:
    sys.path.insert(0, RETINAFACE_DIR)

SCRFD_VENDOR = VENDOR_DIR
if SCRFD_VENDOR not in sys.path:
    sys.path.insert(0, SCRFD_VENDOR)

TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = ["normal", "gay", "hukusu", "rez"]

RETINAFACE_WEIGHTS = (
    "/home/pan/プロジェクト1/15.AIエージェント/SadTalker/gfpgan/weights/"
    "detection_Resnet50_Final.pth"
)
SCRFD_25G_PATH = os.path.join(MODELS_DIR, "scrfd", "scrfd_2.5g.onnx")
SCRFD_34G_PATH = (
    os.path.join(MODELS_DIR, "")
    "scrfd/models/scrfd_34g_gnkps.onnx"
)

DEFAULT_CONFS = [0.15]
RETINA_NMS = 0.4
SCRFD_NMS = 0.45
UNION_IOU_TH = 0.50


# ============================================================
# GT / メトリクス
# ============================================================

def load_gt_mask(json_path: Path, h: int, w: int) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") == "face":
            pts = np.array(shape["points"], dtype=np.float32).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p = pred > 0
    g = gt > 0
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    cov = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = 2 * prec * cov / (prec + cov) if (prec + cov) > 0 else 0.0
    return {"coverage": cov, "precision": prec, "iou": iou, "f1": f1}


def bboxes_to_mask(bboxes: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


# ============================================================
# UNION ロジック
# ============================================================

def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def union_detections(
    dets_a: np.ndarray,  # Nx5 [x1,y1,x2,y2,score]
    dets_b: np.ndarray,  # Mx5
    iou_threshold: float = UNION_IOU_TH,
) -> np.ndarray:
    if len(dets_a) == 0 and len(dets_b) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if len(dets_a) == 0:
        return dets_b.copy()
    if len(dets_b) == 0:
        return dets_a.copy()
    all_dets = np.vstack([dets_a, dets_b])
    order = all_dets[:, 4].argsort()[::-1]
    all_dets = all_dets[order]
    keep = []
    suppressed = set()
    for i in range(len(all_dets)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(all_dets)):
            if j in suppressed:
                continue
            if bbox_iou(all_dets[i, :4], all_dets[j, :4]) > iou_threshold:
                suppressed.add(j)
    return all_dets[keep]


# ============================================================
# RetinaFace
# ============================================================

class RetinaFaceDetector:
    def __init__(self, weights_path: str, device: str = "cuda:0"):
        from models.retinaface import RetinaFace
        from data import cfg_re50

        self.cfg = cfg_re50
        self.nms_th = RETINA_NMS
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        model = RetinaFace(cfg=cfg_re50, phase="test")
        pretrained = torch.load(weights_path, map_location="cpu")
        pretrained = {
            k.split("module.", 1)[-1] if k.startswith("module.") else k: v
            for k, v in pretrained.items()
        }
        model.load_state_dict(pretrained, strict=False)
        self.model = model.to(self.device).eval()

    def detect(self, bgr: np.ndarray, conf_th: float) -> np.ndarray:
        from layers.functions.prior_box import PriorBox
        from utils.nms.py_cpu_nms import py_cpu_nms
        from utils.box_utils import decode

        im_h, im_w = bgr.shape[:2]
        img = np.float32(bgr) - np.array([104.0, 117.0, 123.0], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            loc, conf, _ = self.model(img_tensor)

        scale = torch.Tensor([im_w, im_h, im_w, im_h]).to(self.device)
        priorbox = PriorBox(self.cfg, image_size=(im_h, im_w))
        priors = priorbox.forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        boxes = (boxes * scale).cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        inds = np.where(scores > conf_th)[0]
        if len(inds) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        boxes, scores = boxes[inds], scores[inds]
        order = scores.argsort()[::-1][:5000]
        boxes, scores = boxes[order], scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, self.nms_th)
        return dets[keep]  # Nx5 [x1,y1,x2,y2,score]


# ============================================================
# SCRFD
# ============================================================

class SCRFDDetector:
    def __init__(self, model_path: str):
        from scrfd.pub import SCRFD
        self.det = SCRFD.from_path(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def detect(self, bgr: np.ndarray, conf_th: float) -> np.ndarray:
        from scrfd.schemas import Threshold
        from PIL import Image
        th = Threshold(probability=conf_th, nms=SCRFD_NMS)
        rgb = Image.fromarray(bgr[:, :, ::-1])
        faces = self.det.detect(rgb, threshold=th)
        if not faces:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([
            [float(f.bbox.upper_left.x), float(f.bbox.upper_left.y),
             float(f.bbox.lower_right.x), float(f.bbox.lower_right.y),
             float(f.probability)]
            for f in faces
        ], dtype=np.float32)


# ============================================================
# 評価ループ
# ============================================================

def eval_testset(
    name: str,
    scrfd_25g: SCRFDDetector,
    scrfd_34g: SCRFDDetector,
    retina: RetinaFaceDetector,
    conf_th: float,
    max_frames: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    ts_dir = TESTSET_ROOT / name
    frame_files = sorted(ts_dir.glob("*.jpg"))
    if max_frames:
        frame_files = frame_files[:max_frames]

    model_names = ["SCRFD-2.5g", "SCRFD-34g", "RetinaFace-R50", "UNION(34g+Retina)"]
    per_frame: Dict[str, List] = {k: [] for k in model_names}

    for img_path in tqdm(frame_files, desc=f"  {name} (conf={conf_th})", leave=False):
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

        # 各モデルの検出
        dets_25g = scrfd_25g.detect(bgr, conf_th)
        dets_34g = scrfd_34g.detect(bgr, conf_th)
        dets_retina = retina.detect(bgr, conf_th)

        # UNION: SCRFD-34g + RetinaFace R50
        dets_union = union_detections(dets_34g, dets_retina, UNION_IOU_TH)

        all_dets = {
            "SCRFD-2.5g": dets_25g,
            "SCRFD-34g": dets_34g,
            "RetinaFace-R50": dets_retina,
            "UNION(34g+Retina)": dets_union,
        }

        for det_name, dets in all_dets.items():
            boxes = dets[:, :4] if len(dets) > 0 else np.zeros((0, 4), dtype=np.float32)
            pred = bboxes_to_mask(boxes, h, w) if len(boxes) > 0 else np.zeros((h, w), dtype=np.uint8)
            m = compute_metrics(pred, gt)
            m["detected"] = int(len(boxes) > 0)
            per_frame[det_name].append(m)

    results = {}
    for det_name, frames in per_frame.items():
        if not frames:
            results[det_name] = {"coverage": 0, "precision": 0, "iou": 0, "f1": 0,
                                  "det_rate": 0, "n_frames": 0}
            continue
        results[det_name] = {
            "coverage":  round(float(np.mean([m["coverage"]  for m in frames])), 4),
            "precision": round(float(np.mean([m["precision"] for m in frames])), 4),
            "iou":       round(float(np.mean([m["iou"]       for m in frames])), 4),
            "f1":        round(float(np.mean([m["f1"]        for m in frames])), 4),
            "det_rate":  round(float(np.mean([m["detected"]  for m in frames])), 4),
            "n_frames":  len(frames),
        }
    return results


def weighted_avg(per_ts: Dict, det_name: str) -> Dict[str, float]:
    total_n = sum(v[det_name]["n_frames"] for v in per_ts.values())
    if total_n == 0:
        return {"coverage": 0, "precision": 0, "iou": 0, "f1": 0, "det_rate": 0}
    keys = ["coverage", "precision", "iou", "f1", "det_rate"]
    avg = {}
    for k in keys:
        avg[k] = round(
            sum(v[det_name][k] * v[det_name]["n_frames"]
                for v in per_ts.values()) / total_n, 4
        )
    avg["n_frames"] = total_n
    return avg


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="UNION 検出（SCRFD-34g + RetinaFace R50）の実測評価"
    )
    parser.add_argument(
        "--conf", type=float, nargs="+", default=DEFAULT_CONFS,
        help="評価する信頼度閾値リスト (default: 0.15)",
    )
    parser.add_argument("--out", default="output/eval_union_compare.json")
    parser.add_argument("--testset", nargs="+", default=TESTSETS, choices=TESTSETS)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("UNION 検出評価: SCRFD-2.5g / SCRFD-34g / RetinaFace R50 / UNION")
    print(f"  信頼度閾値: {args.conf}")
    print(f"  テストセット: {args.testset}")
    print(f"  UNION NMS IoU閾値: {UNION_IOU_TH}")
    print("=" * 70)

    # --- モデルロード ---
    print("\n[1] モデルロード中...")
    print("  RetinaFace R50 ...")
    retina = RetinaFaceDetector(RETINAFACE_WEIGHTS)
    print("  SCRFD-2.5g ...")
    scrfd_25g = SCRFDDetector(SCRFD_25G_PATH)
    print("  SCRFD-34g ...")
    scrfd_34g = SCRFDDetector(SCRFD_34G_PATH)
    print("  ロード完了\n")

    model_names = ["SCRFD-2.5g", "SCRFD-34g", "RetinaFace-R50", "UNION(34g+Retina)"]
    all_results = {}
    t0 = time.time()

    for conf_th in args.conf:
        conf_key = f"conf={conf_th:.2f}"
        print(f"\n{'='*70}")
        print(f"  conf_th = {conf_th}")
        print(f"{'='*70}")

        per_ts = {}
        for ts in args.testset:
            per_ts[ts] = eval_testset(ts, scrfd_25g, scrfd_34g, retina, conf_th, args.max_frames)

        wavg = {}
        for det_name in model_names:
            wavg[det_name] = weighted_avg(per_ts, det_name)

        # 表示
        header = f"  {'モデル':25s}  Coverage  Precision  IoU       F1        det_rate"
        print(f"\n  加重平均 (conf={conf_th}):")
        print(header)
        print("  " + "-" * 80)
        for det_name in model_names:
            w = wavg[det_name]
            print(
                f"  {det_name:25s}  {w['coverage']:.4f}    {w['precision']:.4f}     "
                f"{w['iou']:.4f}    {w['f1']:.4f}    {w['det_rate']:.4f}"
            )

        # UNION vs 個別の差分表示
        print(f"\n  UNION vs 個別モデルの差分 (conf={conf_th}):")
        print(f"  {'比較':25s}  Coverage  Precision  IoU       F1")
        print("  " + "-" * 70)
        u = wavg["UNION(34g+Retina)"]
        for det_name in ["SCRFD-2.5g", "SCRFD-34g", "RetinaFace-R50"]:
            w = wavg[det_name]
            dc = u["coverage"] - w["coverage"]
            dp = u["precision"] - w["precision"]
            di = u["iou"] - w["iou"]
            df = u["f1"] - w["f1"]
            print(
                f"  UNION vs {det_name:15s}  {dc:+.4f}    {dp:+.4f}     "
                f"{di:+.4f}    {df:+.4f}"
            )

        all_results[conf_key] = {
            "conf_th": conf_th,
            "per_testset": per_ts,
            "weighted_avg": wavg,
        }

    elapsed = time.time() - t0

    # --- 総合サマリ ---
    print(f"\n\n{'='*70}")
    print("  総合サマリ")
    print(f"{'='*70}")
    for conf_key, data in all_results.items():
        conf_th = data["conf_th"]
        print(f"\n  conf = {conf_th}")
        print(f"  {'モデル':25s}  Coverage  Precision  IoU       F1        det_rate")
        print("  " + "-" * 80)
        for det_name in model_names:
            w = data["weighted_avg"][det_name]
            print(
                f"  {det_name:25s}  {w['coverage']:.4f}    {w['precision']:.4f}     "
                f"{w['iou']:.4f}    {w['f1']:.4f}    {w['det_rate']:.4f}"
            )

    # --- 保存 ---
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {output_path}")
    print(f"総処理時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
