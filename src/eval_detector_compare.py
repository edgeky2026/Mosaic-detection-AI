"""
代替顔検出モデル比較評価スクリプト

SCRFD-2.5g / SCRFD-34g / YuNet / YOLO11n-face / アンサンブル(SCRFD+YuNet) を
testset_2603 (normal/gay/hukusu/rez) で比較評価する。

使用方法:
    python eval_detector_compare.py --out /path/to/result.json
    python eval_detector_compare.py --testset normal gay  # 特定テストセットのみ
    python eval_detector_compare.py --max-frames 100      # デバッグ用

評価指標:
    Coverage (Recall) = GT 顔ピクセルを何割捉えたか（見逃し率の逆）
    Precision         = 予測マスク中の本物顔ピクセル割合
    IoU               = Coverage と Precision の複合指標
    det_rate          = GT が存在するフレームで 1 件以上検出できた割合
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

SCRFD_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
if SCRFD_VENDOR not in sys.path:
    sys.path.insert(0, SCRFD_VENDOR)

TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = ["normal", "gay", "hukusu", "rez"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRFD_25G_PATH = os.path.join(_REPO_ROOT, "models", "scrfd", "scrfd_2.5g.onnx")
SCRFD_34G_PATH = os.path.join(_REPO_ROOT, "models", "scrfd", "scrfd_34g.onnx")
YUNET_PATH     = os.environ.get("YUNET_PATH", "models/yunet_face.onnx")
YOLO_FACE_PATH = os.environ.get("YOLO_FACE_PATH", "models/yolo11n-face.pt")

SCRFD_CONF = 0.15
SCRFD_NMS  = 0.45
YUNET_CONF = 0.6
YUNET_NMS  = 0.3
YOLO_CONF  = 0.25


# ============================================================
# GT マスク生成
# ============================================================

def load_gt_mask(json_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """LabelMe JSON → GT バイナリマスク"""
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
# 評価指標
# ============================================================

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = (pred_mask > 0)
    gt   = (gt_mask > 0)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    coverage  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"coverage": coverage, "precision": precision, "iou": iou}


def bboxes_to_mask(bboxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nx4 [x1,y1,x2,y2] → フルマスク"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in bboxes:
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


# ============================================================
# 検出器の初期化
# ============================================================

def load_scrfd(model_path: str):
    """SCRFD 検出器をロード"""
    from scrfd.pub import SCRFD
    from scrfd.schemas import Threshold
    det = SCRFD.from_path(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    th  = Threshold(probability=SCRFD_CONF, nms=SCRFD_NMS)
    return det, th


def detect_scrfd(det, th, bgr: np.ndarray) -> np.ndarray:
    """SCRFD 検出 → Nx4 [x1,y1,x2,y2]"""
    from PIL import Image
    rgb = Image.fromarray(bgr[:, :, ::-1])
    faces = det.detect(rgb, threshold=th)
    if not faces:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for f in faces:
        x1 = float(f.bbox.upper_left.x)
        y1 = float(f.bbox.upper_left.y)
        x2 = float(f.bbox.lower_right.x)
        y2 = float(f.bbox.lower_right.y)
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32)


def load_yunet(img_h: int, img_w: int):
    """YuNet 検出器をロード (OpenCV FaceDetectorYN)"""
    det = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (img_w, img_h),
        score_threshold=YUNET_CONF,
        nms_threshold=YUNET_NMS,
        top_k=5000,
    )
    return det


def detect_yunet(det, bgr: np.ndarray) -> np.ndarray:
    """YuNet 検出 → Nx4 [x1,y1,x2,y2]"""
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    faces = det.detect(bgr)
    if faces[1] is None:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for face in faces[1]:
        x, y, fw, fh = face[0], face[1], face[2], face[3]
        boxes.append([x, y, x + fw, y + fh])
    return np.array(boxes, dtype=np.float32)


def load_yolo_face():
    """YOLO11n-face ロード (Ultralytics)"""
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_FACE_PATH)
        return model
    except Exception as e:
        print(f"[WARN] YOLO face 読み込み失敗: {e}")
        return None


def detect_yolo_face(model, bgr: np.ndarray) -> np.ndarray:
    """YOLO 顔検出 → Nx4 [x1,y1,x2,y2]"""
    if model is None:
        return np.zeros((0, 4), dtype=np.float32)
    results = model(bgr, conf=YOLO_CONF, verbose=False)
    if not results or results[0].boxes is None:
        return np.zeros((0, 4), dtype=np.float32)
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    return xyxy.astype(np.float32) if len(xyxy) > 0 else np.zeros((0, 4), dtype=np.float32)


# ============================================================
# テストセット評価
# ============================================================

def eval_testset(
    name: str,
    detectors: Dict,
    max_frames: Optional[int] = None,
) -> Dict:
    """
    1 テストセットを全検出器で評価する。

    Returns:
        {detector_name: {coverage, precision, iou, det_rate, n_frames}}
    """
    ts_dir = TESTSET_ROOT / name
    frame_files = sorted([f for f in ts_dir.iterdir() if f.suffix == ".jpg"])
    if max_frames:
        frame_files = frame_files[:max_frames]

    # 各検出器の per-frame metrics
    per_frame = {k: [] for k in detectors}

    first_img = cv2.imread(str(frame_files[0]))
    img_h, img_w = first_img.shape[:2]

    # YuNet は画像サイズに合わせた初期化が必要
    if "yunet" in detectors and detectors["yunet"] is not None:
        if isinstance(detectors["yunet"], str) and detectors["yunet"] == "lazy":
            detectors["yunet"] = load_yunet(img_h, img_w)

    for img_path in tqdm(frame_files, desc=f"  {name}", leave=False):
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt_mask = load_gt_mask(json_path, h, w)

        if gt_mask.sum() == 0:
            # GT が空のフレームはスキップ
            continue

        # 各検出器で検出 → メトリクス計算
        det_results = {}
        for det_name, det_obj in detectors.items():
            if det_name.startswith("scrfd"):
                key = det_name.split("_")[1]  # "25g" or "34g"
                det_fn, th = det_obj
                boxes = detect_scrfd(det_fn, th, bgr)
            elif det_name == "yunet":
                boxes = detect_yunet(det_obj, bgr)
            elif det_name == "yolo11n_face":
                boxes = detect_yolo_face(det_obj, bgr)
            elif det_name == "ensemble_25g_yunet":
                # SCRFD-2.5g OR YuNet のアンサンブル
                det_25g_fn, th_25g = detectors["scrfd_25g"]
                boxes_s = detect_scrfd(det_25g_fn, th_25g, bgr)
                boxes_y = detect_yunet(detectors["yunet"], bgr)
                boxes = np.vstack([boxes_s, boxes_y]) if (len(boxes_s) > 0 and len(boxes_y) > 0) \
                        else (boxes_s if len(boxes_s) > 0 else boxes_y)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)

            pred_mask = bboxes_to_mask(boxes, h, w) if len(boxes) > 0 \
                        else np.zeros((h, w), dtype=np.uint8)
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics["detected"] = int(len(boxes) > 0)
            det_results[det_name] = metrics

        for det_name, m in det_results.items():
            per_frame[det_name].append(m)

    # 集計
    results = {}
    for det_name, frames in per_frame.items():
        if not frames:
            results[det_name] = {"coverage": 0.0, "precision": 0.0, "iou": 0.0,
                                  "det_rate": 0.0, "n_frames": 0}
            continue
        cov  = float(np.mean([m["coverage"]  for m in frames]))
        prec = float(np.mean([m["precision"] for m in frames]))
        iou  = float(np.mean([m["iou"]       for m in frames]))
        det_rate = float(np.mean([m["detected"] for m in frames]))
        results[det_name] = {
            "coverage": round(cov, 4),
            "precision": round(prec, 4),
            "iou": round(iou, 4),
            "det_rate": round(det_rate, 4),
            "n_frames": len(frames),
        }
    return results


def weighted_avg(per_testset: Dict, det_name: str) -> Dict:
    """全テストセットの加重平均を計算"""
    total_n = sum(per_testset[ts][det_name]["n_frames"] for ts in per_testset)
    if total_n == 0:
        return {"coverage": 0.0, "precision": 0.0, "iou": 0.0, "det_rate": 0.0}
    cov  = sum(per_testset[ts][det_name]["coverage"]  * per_testset[ts][det_name]["n_frames"] for ts in per_testset) / total_n
    prec = sum(per_testset[ts][det_name]["precision"] * per_testset[ts][det_name]["n_frames"] for ts in per_testset) / total_n
    iou  = sum(per_testset[ts][det_name]["iou"]       * per_testset[ts][det_name]["n_frames"] for ts in per_testset) / total_n
    dr   = sum(per_testset[ts][det_name]["det_rate"]  * per_testset[ts][det_name]["n_frames"] for ts in per_testset) / total_n
    return {
        "coverage": round(cov, 4),
        "precision": round(prec, 4),
        "iou": round(iou, 4),
        "det_rate": round(dr, 4),
        "n_frames": total_n,
    }


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="代替顔検出モデル比較評価")
    parser.add_argument("--out", default="output/eval_detector_compare.json")
    parser.add_argument("--testset", nargs="+", default=TESTSETS,
                        choices=TESTSETS, help="評価するテストセット")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="デバッグ用: 各テストセットのフレーム上限")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="YOLO 評価をスキップ（速度優先時）")
    parser.add_argument("--skip-34g", action="store_true",
                        help="SCRFD-34g をスキップ")
    args = parser.parse_args()

    print("=" * 60)
    print("代替顔検出モデル比較評価")
    print("=" * 60)

    # --- 検出器ロード ---
    print("\n[1/5] 検出器ロード中...")

    print("  SCRFD-2.5g ...")
    det_25g, th_25g = load_scrfd(SCRFD_25G_PATH)
    print("  SCRFD-2.5g OK")

    det_34g, th_34g = None, None
    if not args.skip_34g and os.path.exists(SCRFD_34G_PATH):
        print("  SCRFD-34g ...")
        det_34g, th_34g = load_scrfd(SCRFD_34G_PATH)
        print("  SCRFD-34g OK")
    else:
        print("  SCRFD-34g スキップ (--skip-34g または ファイル未存在)")

    print("  YuNet ...")
    # YuNet はフレームサイズ依存 → 評価ループで遅延初期化 ("lazy" マーカー)
    yunet_det = "lazy"  # 実際の初期化は eval_testset 内で行う
    print("  YuNet OK (遅延初期化)")

    yolo_det = None
    if not args.skip_yolo and os.path.exists(YOLO_FACE_PATH):
        print("  YOLO11n-face ...")
        yolo_det = load_yolo_face()
        print(f"  YOLO11n-face {'OK' if yolo_det else 'FAILED'}")
    else:
        print("  YOLO11n-face スキップ")

    # 検出器辞書
    detectors = {
        "scrfd_25g": (det_25g, th_25g),
        "yunet": yunet_det,
        "ensemble_25g_yunet": "computed_from_components",
    }
    if det_34g is not None:
        detectors["scrfd_34g"] = (det_34g, th_34g)
    if yolo_det is not None:
        detectors["yolo11n_face"] = yolo_det

    # アンサンブルキーは detect 内で動的に処理するため、別途フラグ管理
    # ensemble は scrfd_25g と yunet の両方が必要

    print(f"\n[2/5] 評価対象テストセット: {args.testset}")
    print(f"      最大フレーム数: {'全件' if args.max_frames is None else args.max_frames}")

    # --- 評価実行 ---
    t_start = time.time()
    per_testset_results = {}

    for ts in args.testset:
        print(f"\n  テストセット: {ts}")
        # YuNet を各テストセットで再ロード（解像度確認）
        # 最初のフレームで解像度を取得して初期化
        ts_dir = TESTSET_ROOT / ts
        first_img_path = sorted(ts_dir.glob("*.jpg"))[0]
        first_img = cv2.imread(str(first_img_path))
        h0, w0 = first_img.shape[:2]
        detectors["yunet"] = load_yunet(h0, w0)

        results = eval_testset(ts, detectors, max_frames=args.max_frames)
        per_testset_results[ts] = results
        # サマリ表示
        for det_name, m in results.items():
            print(f"    {det_name:30s} cov={m['coverage']:.4f}  prec={m['precision']:.4f}  iou={m['iou']:.4f}  det_rate={m['det_rate']:.4f}  n={m['n_frames']}")

    elapsed = time.time() - t_start

    # --- 加重平均 ---
    print("\n[3/5] 加重平均計算...")
    det_names = list(per_testset_results[args.testset[0]].keys())
    weighted_results = {}
    for det_name in det_names:
        weighted_results[det_name] = weighted_avg(per_testset_results, det_name)

    print("\n  ===== 加重平均 =====")
    for det_name, m in weighted_results.items():
        print(f"  {det_name:30s} cov={m['coverage']:.4f}  prec={m['precision']:.4f}  iou={m['iou']:.4f}  det_rate={m['det_rate']:.4f}")

    # --- 保存 ---
    output = {
        "testsets": args.testset,
        "max_frames": args.max_frames,
        "elapsed_sec": round(elapsed, 1),
        "params": {
            "scrfd_conf": SCRFD_CONF,
            "yunet_conf": YUNET_CONF,
            "yunet_nms": YUNET_NMS,
            "yolo_conf": YOLO_CONF,
        },
        "per_testset": per_testset_results,
        "weighted_avg": weighted_results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[4/5] 結果保存: {args.out}")
    print(f"[5/5] 完了 ({elapsed:.1f}秒)")


if __name__ == "__main__":
    main()
