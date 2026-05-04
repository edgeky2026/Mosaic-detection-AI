"""
RetinaFace (ResNet50) 顔検出評価スクリプト

biubug6/Pytorch_Retinaface（vendor 内 git clone）を使用し、
testset_2603（normal/gay/hukusu/rez）で SCRFD-2.5g / SCRFD-34g と比較評価する。

目的:
  アップロード時検知①（FP 極小化設計）に最適な検出器を選定するため、
  RetinaFace R50 の実測精度を既存 SCRFD モデルと比較する。

使用方法:
    python eval_retinaface.py
    python eval_retinaface.py --conf 0.5 --out result.json   # 本番想定 conf_th
    python eval_retinaface.py --max-frames 50                # クイックテスト
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
import torch
from tqdm import tqdm

# --- パス設定 ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

VENDOR_DIR = os.path.dirname(SRC_DIR) + "/vendor"
RETINAFACE_DIR = VENDOR_DIR + "/Pytorch_Retinaface"
if RETINAFACE_DIR not in sys.path:
    sys.path.insert(0, RETINAFACE_DIR)

SCRFD_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
if SCRFD_VENDOR not in sys.path:
    sys.path.insert(0, SCRFD_VENDOR)

# テストセット
TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = ["normal", "gay", "hukusu", "rez"]

# モデルパス
_REPO_ROOT_EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETINAFACE_WEIGHTS = os.environ.get(
    "RETINAFACE_WEIGHTS", "models/retinaface/detection_Resnet50_Final.pth"
)
SCRFD_25G_PATH = os.path.join(_REPO_ROOT_EVAL, "models", "scrfd", "scrfd_2.5g.onnx")
SCRFD_34G_PATH = os.path.join(_REPO_ROOT_EVAL, "models", "scrfd", "scrfd_34g.onnx")

SCRFD_CONF = 0.15  # 評価用（論文比較）
SCRFD_NMS  = 0.45
RETINA_NMS = 0.4


# ============================================================
# GT マスク
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


# ============================================================
# 評価指標
# ============================================================

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p = pred > 0
    g = gt > 0
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    return {
        "coverage":  tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "iou":       tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0,
    }


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
# RetinaFace 検出器
# ============================================================

class RetinaFaceDetector:
    """biubug6/Pytorch_Retinaface の RetinaFace R50 ラッパー"""

    def __init__(self, weights_path: str, conf_th: float = 0.5, device: str = "cuda:0"):
        from models.retinaface import RetinaFace
        from data import cfg_re50

        self.cfg = cfg_re50
        self.conf_th = conf_th
        self.nms_th = RETINA_NMS
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        model = RetinaFace(cfg=cfg_re50, phase="test")
        pretrained = torch.load(weights_path, map_location="cpu")
        # 'module.' prefix を除去
        pretrained = {
            k.split("module.", 1)[-1] if k.startswith("module.") else k: v
            for k, v in pretrained.items()
        }
        model.load_state_dict(pretrained, strict=False)
        self.model = model.to(self.device).eval()

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """
        BGR 画像から顔を検出する。

        Returns:
            Nx4 [x1, y1, x2, y2] の numpy 配列（0 件の場合は shape=(0,4)）
        """
        from layers.functions.prior_box import PriorBox
        from utils.nms.py_cpu_nms import py_cpu_nms
        from utils.box_utils import decode

        im_h, im_w = bgr.shape[:2]

        # 前処理（BGR mean subtraction）
        img = np.float32(bgr) - np.array([104.0, 117.0, 123.0], dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            loc, conf, _ = self.model(img_tensor)

        # デコード
        scale = torch.Tensor([im_w, im_h, im_w, im_h]).to(self.device)
        priorbox = PriorBox(self.cfg, image_size=(im_h, im_w))
        priors = priorbox.forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, self.cfg["variance"])
        boxes = (boxes * scale).cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # 信頼度フィルタ
        inds = np.where(scores > self.conf_th)[0]
        if len(inds) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes = boxes[inds]
        scores = scores[inds]

        # スコア降順ソート → top_k=5000
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, self.nms_th)
        dets = dets[keep]

        return dets[:, :4]


# ============================================================
# SCRFD 検出器
# ============================================================

def load_scrfd(model_path: str, conf_th: float):
    from scrfd.pub import SCRFD
    from scrfd.schemas import Threshold
    det = SCRFD.from_path(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    th  = Threshold(probability=conf_th, nms=SCRFD_NMS)
    return det, th


def detect_scrfd(det, th, bgr: np.ndarray) -> np.ndarray:
    from PIL import Image
    rgb = Image.fromarray(bgr[:, :, ::-1])
    faces = det.detect(rgb, threshold=th)
    if not faces:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array([
        [float(f.bbox.upper_left.x), float(f.bbox.upper_left.y),
         float(f.bbox.lower_right.x), float(f.bbox.lower_right.y)]
        for f in faces
    ], dtype=np.float32)


# ============================================================
# テストセット評価
# ============================================================

def eval_one_testset(
    name: str,
    detectors: Dict,
    max_frames: Optional[int] = None,
) -> Dict:
    ts_dir = TESTSET_ROOT / name
    frame_files = sorted(ts_dir.glob("*.jpg"))
    if max_frames:
        frame_files = frame_files[:max_frames]

    per_frame: Dict[str, List] = {k: [] for k in detectors}

    for img_path in tqdm(frame_files, desc=f"  {name}", leave=False):
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

        for det_name, det_obj in detectors.items():
            if det_name.startswith("retinaface"):
                boxes = det_obj.detect(bgr)
            elif det_name.startswith("scrfd"):
                fn, th = det_obj
                boxes = detect_scrfd(fn, th, bgr)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

            pred = bboxes_to_mask(boxes, h, w) if len(boxes) > 0 else np.zeros((h, w), dtype=np.uint8)
            m = compute_metrics(pred, gt)
            m["detected"] = int(len(boxes) > 0)
            per_frame[det_name].append(m)

    results = {}
    for det_name, frames in per_frame.items():
        if not frames:
            results[det_name] = {"coverage": 0.0, "precision": 0.0, "iou": 0.0,
                                  "det_rate": 0.0, "n_frames": 0}
            continue
        results[det_name] = {
            "coverage":  round(float(np.mean([m["coverage"]  for m in frames])), 4),
            "precision": round(float(np.mean([m["precision"] for m in frames])), 4),
            "iou":       round(float(np.mean([m["iou"]       for m in frames])), 4),
            "det_rate":  round(float(np.mean([m["detected"]  for m in frames])), 4),
            "n_frames":  len(frames),
        }
    return results


def weighted_avg(per_ts: Dict, det_name: str) -> Dict:
    total_n = sum(v[det_name]["n_frames"] for v in per_ts.values())
    if total_n == 0:
        return {"coverage": 0.0, "precision": 0.0, "iou": 0.0, "det_rate": 0.0}
    keys = ["coverage", "precision", "iou", "det_rate"]
    avg = {}
    for k in keys:
        avg[k] = round(
            sum(v[det_name][k] * v[det_name]["n_frames"] for v in per_ts.values()) / total_n, 4
        )
    avg["n_frames"] = total_n
    return avg


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RetinaFace R50 vs SCRFD 比較評価")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="RetinaFace 信頼度閾値（デフォルト=0.3 評価用, 本番=0.5）")
    parser.add_argument("--scrfd-conf", type=float, default=SCRFD_CONF,
                        help="SCRFD 信頼度閾値")
    parser.add_argument("--out", default="output/eval_retinaface.json")
    parser.add_argument("--testset", nargs="+", default=TESTSETS, choices=TESTSETS)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    print("=" * 65)
    print("RetinaFace R50 vs SCRFD-2.5g / SCRFD-34g 比較評価")
    print(f"  RetinaFace conf_th = {args.conf}  |  SCRFD conf_th = {args.scrfd_conf}")
    print("=" * 65)

    # --- 検出器ロード ---
    print("\n[1/4] 検出器ロード中...")

    print("  RetinaFace R50 ...")
    retina_det = RetinaFaceDetector(RETINAFACE_WEIGHTS, conf_th=args.conf)
    print(f"  RetinaFace R50 OK (conf_th={args.conf})")

    print("  SCRFD-2.5g ...")
    scrfd_25g = load_scrfd(SCRFD_25G_PATH, args.scrfd_conf)
    print("  SCRFD-2.5g OK")

    print("  SCRFD-34g ...")
    scrfd_34g = load_scrfd(SCRFD_34G_PATH, args.scrfd_conf)
    print("  SCRFD-34g OK")

    # アンサンブル用に RetinaFace(0.5) も追加で評価
    retina_det_high = RetinaFaceDetector(RETINAFACE_WEIGHTS, conf_th=0.5)

    detectors = {
        "scrfd_25g":          scrfd_25g,
        "scrfd_34g":          scrfd_34g,
        f"retinaface_r50_c{int(args.conf*100):02d}": retina_det,
        "retinaface_r50_c50": retina_det_high,
    }

    print(f"\n[2/4] 評価: {args.testset}, max_frames={args.max_frames or '全件'}")
    t0 = time.time()
    per_ts = {}

    for ts in args.testset:
        print(f"\n  testset: {ts}")
        per_ts[ts] = eval_one_testset(ts, detectors, max_frames=args.max_frames)
        for det_name, m in per_ts[ts].items():
            print(f"    {det_name:35s} cov={m['coverage']:.4f}  prec={m['precision']:.4f}"
                  f"  iou={m['iou']:.4f}  det={m['det_rate']:.4f}  n={m['n_frames']}")

    elapsed = time.time() - t0

    # 加重平均
    print("\n[3/4] 加重平均")
    print(f"  {'検出器':35s} Coverage   Precision  IoU        det_rate")
    print("  " + "-" * 70)
    wavg = {}
    for det_name in detectors:
        w = weighted_avg(per_ts, det_name)
        wavg[det_name] = w
        print(f"  {det_name:35s} {w['coverage']:.4f}     {w['precision']:.4f}     "
              f"{w['iou']:.4f}     {w['det_rate']:.4f}")

    # 保存
    output = {
        "conf_retinaface": args.conf,
        "conf_scrfd": args.scrfd_conf,
        "testsets": args.testset,
        "max_frames": args.max_frames,
        "elapsed_sec": round(elapsed, 1),
        "per_testset": per_ts,
        "weighted_avg": wavg,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[4/4] 保存: {args.out}  ({elapsed:.1f}秒)")


if __name__ == "__main__":
    main()
