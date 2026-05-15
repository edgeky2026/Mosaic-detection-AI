"""
性器モザイク漏れ検知パイプライン — testset_2603 評価スクリプト

testset_2603 (normal / gay / hukusu / rez) の各フレームに対して
Fine-tuned GroundingDino を実行し、GT ポリゴンマスクと比較して
Coverage / Precision / IoU を算出する。

GT ラベル（性器関連）: vagina_wide, penis, anus_insertion

使用方法:
  /home/pan/miniforge3/envs/ml/bin/python eval_genital.py
  /home/pan/miniforge3/envs/ml/bin/python eval_genital.py --subset normal
  /home/pan/miniforge3/envs/ml/bin/python eval_genital.py --genital-only
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
from dino_checkpoint_utils import get_default_dino_checkpoint, resolve_dino_checkpoint

# --- パス設定 ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(REPO_DIR, "models")
VENDOR_DIR = os.path.join(REPO_DIR, "vendor")

sys.path.insert(0, SRC_DIR)
# vendor/grounding_dino/ を追加 → "from groundingdino import _C" が通る（CUDA ops）
_gdino_pkg = os.path.join(VENDOR_DIR, "grounding_dino")
if os.path.isdir(_gdino_pkg) and _gdino_pkg not in sys.path:
    sys.path.insert(0, _gdino_pkg)
# VENDOR_DIR 自体も追加 → "from grounding_dino.groundingdino..." が通る
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

# テストセットのパス設定
TESTSET_ROOT = Path(os.environ.get("TESTSET_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/testset_2603"))
TESTSETS = {
    "normal": TESTSET_ROOT / "normal",
    "gay":    TESTSET_ROOT / "gay",
    "hukusu": TESTSET_ROOT / "hukusu",
    "rez":    TESTSET_ROOT / "rez",
}

# GT ラベル
GENITAL_LABELS = {"vagina_wide", "penis", "anus_insertion"}

# デフォルトチェックポイント
DEFAULT_DINO_CHECKPOINT = get_default_dino_checkpoint()
DEFAULT_DINO_CONFIG = os.path.join(MODELS_DIR, "gdino", "cfg_odvg.py")
GENITAL_TEXT_PROMPT = "vagina . penis . anus ."
BOX_THRESHOLD = 0.18    # 本番パイプライン設定値（施策H: Epoch 3）
TEXT_THRESHOLD = 0.15
MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.30
MAX_ASPECT_RATIO = 4.0      # 施策J: 極端に細長いBBoxを体部位FPとして棄却


def aggregate_weighted_subset_metrics(subset_results: List[Dict]) -> Dict[str, float]:
    """評価フレーム数で重み付けした全体平均を返す。"""
    total_eval = sum(result["n_evaluated"] for result in subset_results)
    if total_eval == 0:
        return {"coverage": 0.0, "precision": 0.0, "iou": 0.0}

    weighted = {}
    for key in ("coverage", "precision", "iou"):
        weighted[key] = sum(
            result["metrics"][key] * result["n_evaluated"]
            for result in subset_results
        ) / total_eval
    return weighted


# ============================================================
# GT マスク生成
# ============================================================

def load_gt_mask(
    json_path: Path, img_h: int, img_w: int, labels: set
) -> np.ndarray:
    """
    LabelMe JSON のポリゴンアノテーションから GT バイナリマスクを生成する。

    Args:
        json_path: LabelMe JSON ファイルのパス
        img_h, img_w: 画像サイズ
        labels: 対象ラベルセット

    Returns:
        (H, W) uint8 マスク (1=対象領域, 0=背景)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") in labels:
            pts = np.array(shape["points"], dtype=np.float32)
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# 評価指標計算
# ============================================================

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Coverage / Precision / IoU を計算する"""
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask > 0).astype(np.uint8)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt.astype(bool))))
    fn = int(np.sum((~pred.astype(bool)) & gt))

    coverage  = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"coverage": coverage, "precision": precision, "iou": iou}


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    """フレームリストの metrics を平均する"""
    if not metric_list:
        return {"coverage": 0.0, "precision": 0.0, "iou": 0.0}
    coverage  = float(np.mean([m["coverage"]  for m in metric_list]))
    precision = float(np.mean([m["precision"] for m in metric_list]))
    iou       = float(np.mean([m["iou"]       for m in metric_list]))
    return {"coverage": coverage, "precision": precision, "iou": iou}


# ============================================================
# GroundingDino ロード・推論
# ============================================================

def load_grounding_dino(dino_config: str, dino_checkpoint: str):
    """Fine-tuned GroundingDino モデルをロード"""
    try:
        from groundingdino.util.inference import load_model
    except ImportError as e:
        raise ImportError(
            f"GroundingDino が見つかりません: {e}\n"
            f"  cd {VENDOR_DIR} && pip install --no-build-isolation -e grounding_dino"
        )
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(dino_config, dino_checkpoint, device=device)
    model = model.to(device)
    return model, device


def detect_genitals_frame(
    model,
    device: str,
    bgr_image: np.ndarray,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> np.ndarray:
    """
    1フレームに対してGroundingDinoで性器検出し、予測マスク(H,W)を返す。
    """
    from PIL import Image as PILImage
    import torch
    import torchvision.transforms as T
    from torchvision.ops import box_convert
    from groundingdino.util.inference import predict

    h, w = bgr_image.shape[:2]
    rgb_pil = PILImage.fromarray(bgr_image[:, :, ::-1])

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(rgb_pil)

    with torch.no_grad():
        boxes, logits, _ = predict(
            model=model,
            image=image_tensor,
            caption=GENITAL_TEXT_PROMPT,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    pred_mask = np.zeros((h, w), dtype=np.uint8)

    if len(boxes) == 0:
        return pred_mask

    boxes_xyxy = box_convert(
        boxes=boxes * torch.tensor([w, h, w, h], dtype=torch.float32),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).numpy()

    for box, score in zip(boxes_xyxy, logits.numpy()):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if box_area < w * h * MIN_BOX_AREA_RATIO:
            continue
        if box_area > w * h * MAX_BOX_AREA_RATIO:
            continue
        # 施策J: アスペクト比フィルタ
        bw = x2 - x1
        bh = y2 - y1
        if bw > 0 and bh > 0:
            ar = max(bw / bh, bh / bw)
            if ar > MAX_ASPECT_RATIO:
                continue
        pred_mask[y1:y2, x1:x2] = 1

    return pred_mask


# ============================================================
# 評価メインループ
# ============================================================

def evaluate_subset(
    subset_name: str,
    subset_dir: Path,
    model,
    device: str,
    genital_only: bool = False,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> Dict:
    """
    1サブセットを評価する。

    Returns:
        {"n_frames": int, "n_gt_frames": int, "metrics": {"coverage":, "precision":, "iou":}}
    """
    frame_files = sorted(subset_dir.glob("*.jpg"))
    all_metrics = []
    n_gt_frames = 0
    skipped = 0

    for img_path in frame_files:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            skipped += 1
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            continue

        h, w = frame.shape[:2]
        gt_mask = load_gt_mask(json_path, h, w, GENITAL_LABELS)

        has_gt = gt_mask.sum() > 0

        if genital_only and not has_gt:
            continue

        if not has_gt:
            # GT なしフレーム → pred なければ正解 (precision=1, coverage=1, iou=1)
            # ただし性器 GT なしフレームは評価から除外するのが適切
            # → genital_only=False でも GT なしフレームはスキップ
            continue

        n_gt_frames += 1

        try:
            pred_mask = detect_genitals_frame(model, device, frame,
                                                box_threshold=box_threshold,
                                                text_threshold=text_threshold)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            skipped += 1
            continue

        metrics = compute_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)

    agg = aggregate_metrics(all_metrics)
    return {
        "n_frames": len(frame_files),
        "n_gt_frames": n_gt_frames,
        "n_evaluated": len(all_metrics),
        "n_skipped": skipped,
        "metrics": agg,
    }


def main():
    parser = argparse.ArgumentParser(description="性器モザイク検知 評価スクリプト")
    parser.add_argument("--dino-checkpoint", default=get_default_dino_checkpoint())
    parser.add_argument("--dino-config", default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD)
    parser.add_argument(
        "--subset", choices=["normal", "gay", "hukusu", "rez", "all"],
        default="all"
    )
    parser.add_argument(
        "--genital-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="性器GT を持つフレームのみ評価（デフォルト: ON）"
    )
    parser.add_argument("--output", default=None, help="結果JSON出力先")
    args = parser.parse_args()

    print("=" * 60)
    print("性器モザイク検知 評価 (testset_2603)")
    print("=" * 60)
    print(f"  checkpoint: {args.dino_checkpoint}")
    print(f"  config    : {args.dino_config}")
    print(f"  box_thr   : {args.box_threshold}")
    print(f"  text_thr  : {args.text_threshold}")
    print(f"  subset    : {args.subset}")

    args.dino_checkpoint = resolve_dino_checkpoint(args.dino_checkpoint)

    if not os.path.isfile(args.dino_checkpoint):
        print(f"\n[ERROR] checkpoint not found: {args.dino_checkpoint}")
        sys.exit(1)

    print("\n[1/2] モデルロード中...")
    t0 = time.time()
    model, device = load_grounding_dino(args.dino_config, args.dino_checkpoint)
    print(f"      device={device}, ロード時間={time.time()-t0:.1f}s")

    subsets_to_run = (
        list(TESTSETS.keys()) if args.subset == "all" else [args.subset]
    )

    print("\n[2/2] 評価中...")
    results = {}
    all_frame_metrics = []

    for subset_name in subsets_to_run:
        subset_dir = TESTSETS.get(subset_name)
        if subset_dir is None or not subset_dir.exists():
            print(f"  [SKIP] {subset_name}: ディレクトリが見つかりません")
            continue

        print(f"\n  [{subset_name}]", end="", flush=True)
        t1 = time.time()
        subset_result = evaluate_subset(
            subset_name, subset_dir, model, device,
            genital_only=args.genital_only,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        elapsed = time.time() - t1

        m = subset_result["metrics"]
        n_eval = subset_result["n_evaluated"]
        n_gt   = subset_result["n_gt_frames"]
        print(
            f"  n_GT={n_gt}, n_eval={n_eval}"
            f"  Coverage={m['coverage']:.3f}  Precision={m['precision']:.3f}  IoU={m['iou']:.3f}"
            f"  ({elapsed:.0f}s)"
        )

        results[subset_name] = subset_result

    # 全サブセット合算
    overall = aggregate_weighted_subset_metrics(list(results.values()))

    total_frames = sum(r["n_frames"] for r in results.values())
    total_gt = sum(r["n_gt_frames"] for r in results.values())
    total_eval = sum(r["n_evaluated"] for r in results.values())

    print("\n" + "=" * 60)
    print("全体結果 (性器GT保有フレームのみ評価)")
    print(f"  合計フレーム     : {total_frames}")
    print(f"  性器GT保有       : {total_gt}")
    print(f"  評価フレーム数   : {total_eval}")
    print(f"  Coverage         : {overall['coverage']:.3f}")
    print(f"  Precision        : {overall['precision']:.3f}")
    print(f"  IoU              : {overall['iou']:.3f}")
    print("=" * 60)

    if args.output:
        out = {
            "config": {
                "checkpoint": args.dino_checkpoint,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
            },
            "overall": {
                "total_frames": total_frames,
                "gt_frames": total_gt,
                "evaluated_frames": total_eval,
                "metrics": overall,
            },
            "subsets": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n結果保存: {args.output}")

    return overall


if __name__ == "__main__":
    main()
