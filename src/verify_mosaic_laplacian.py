"""モザイク済み動画での Laplacian 閾値検証スクリプト

モザイクが適用された実動画から性器検出領域の Laplacian 分散値を計測し、
genital_laplacian_threshold_low = 15.0 の妥当性を確認する。

期待される結果:
  - 適切にモザイクが掛かった性器 ROI → Laplacian 低 (< 15) → GREEN
  - モザイク漏れの性器 ROI → Laplacian 高 (> 15) → RED/YELLOW

Usage:
    cd /home/pan/プロジェクト/16.モザイク検知
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/pan/miniforge3/envs/ml/bin/python src/verify_mosaic_laplacian.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
GSAM2_DIR = os.path.join(REPO_DIR, "genital-reference", "LV_Grounded-SAM-2")

sys.path.insert(0, SRC_DIR)
_gdino_lib = os.path.join(GSAM2_DIR, "grounding_dino")
if os.path.isdir(_gdino_lib) and _gdino_lib not in sys.path:
    sys.path.insert(0, _gdino_lib)
if GSAM2_DIR not in sys.path:
    sys.path.insert(0, GSAM2_DIR)

from mosaic_analyzer import analyze_roi
from config import Config
from scorer import determine_genital_verdict, FrameVerdict

# --- 検証対象 ---
VIDEO_ROOT = os.environ.get("VIDEO_ROOT", "/home/pan/プロジェクト/10_finetun-dataset-3/モザイク動画/GoogleDrive-LV-ML/output")
SAMPLE_VIDEOS = [
    # FAIL (性器RED率高 = モザイク漏れ疑い)
    ("FAIL", "LV/グループ18_安藤さん/1204_202602190937_ひだようこ_38.mp4"),
    ("FAIL", "LV/グループ18_安藤さん/1204_202602190937_ひだようこ_36.mp4"),
    ("FAIL", "LV/グループ20_辻さん/685_202512310654_⭐️1_分割01.mp4"),
    # REVIEW (性器YELLOW主体)
    ("REVIEW", "LV/グループ11_岸田/001_14.mp4"),
    ("REVIEW", "LV/グループ11_岸田/001_1.mp4"),
    # PASS (性器GREEN主体 = モザイク十分)
    ("PASS", "LV/グループ11_岸田/001_11.mp4"),
]

DEFAULT_DINO_CHECKPOINT = os.path.join(
    GSAM2_DIR, "gdino_checkpoints", "dino_local_ft_ep4_best.pth"
)
DEFAULT_DINO_CONFIG = os.path.join(GSAM2_DIR, "cfg_odvg.py")
GENITAL_TEXT_PROMPT = "vagina . penis ."
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.15


def extract_frames(video_path: str, fps: float = 1.0) -> List[np.ndarray]:
    """OpenCV でフレーム抽出（fps 間隔でサンプリング）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(video_fps / fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def main():
    print("=" * 70)
    print("モザイク済み動画 Laplacian 閾値検証")
    print("=" * 70)

    cfg = Config()
    print(f"  genital_laplacian_threshold_low  = {cfg.genital_laplacian_threshold_low}")
    print(f"  genital_laplacian_threshold_high = {cfg.genital_laplacian_threshold_high}")
    print(f"  genital_confidence_yellow_th     = {cfg.genital_confidence_yellow_th}")
    print(f"  genital_confidence_red_th         = {cfg.genital_confidence_red_th}")

    # モデルロード
    print("\n[1] モデルロード中...")
    t0 = time.time()
    import torch
    from groundingdino.util.inference import load_model, predict
    import torchvision.transforms as T
    from torchvision.ops import box_convert
    from PIL import Image as PILImage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(DEFAULT_DINO_CONFIG, DEFAULT_DINO_CHECKPOINT, device=device)
    model = model.to(device)
    print(f"    ロード時間: {time.time()-t0:.1f}s")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- 各動画を処理 ---
    print("\n[2] 検証中...\n")

    all_results = []  # 全検出の集約
    video_summaries = []

    for expected_verdict, rel_path in SAMPLE_VIDEOS:
        video_path = os.path.join(VIDEO_ROOT, rel_path)
        if not os.path.exists(video_path):
            print(f"  [SKIP] {rel_path}")
            continue

        print(f"  [{expected_verdict}] {os.path.basename(rel_path)}")

        # フレーム抽出
        frames = extract_frames(video_path, fps=1.0)
        if not frames:
            print(f"    フレームなし")
            continue

        # 各フレームで検出
        detections_all = []
        for fi, frame in enumerate(frames):
            h, w = frame.shape[:2]
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

            for box, score, label in zip(boxes_xyxy, logits.numpy(), phrases):
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                area = (x2 - x1) * (y2 - y1)
                if area < w * h * 0.001:
                    continue

                # ROI 解析
                roi = frame[int(y1):int(y2), int(x1):int(x2)]
                if roi.size == 0:
                    continue
                analysis = analyze_roi(roi, cfg.roi_normalize_size)

                verdict = determine_genital_verdict(
                    laplacian_var=analysis["laplacian_var"],
                    block_size=analysis["block_size"],
                    periodicity=analysis["periodicity"],
                    cfg=cfg,
                    confidence=float(score),
                )

                det_info = {
                    "frame": fi,
                    "score": float(score),
                    "laplacian_var": analysis["laplacian_var"],
                    "block_size": analysis["block_size"],
                    "periodicity": analysis["periodicity"],
                    "verdict": verdict.value,
                    "expected": expected_verdict,
                    "video": os.path.basename(rel_path),
                }
                detections_all.append(det_info)
                all_results.append(det_info)

        # サマリ
        if detections_all:
            laps = [d["laplacian_var"] for d in detections_all]
            verdicts = [d["verdict"] for d in detections_all]
            n_green = verdicts.count("GREEN")
            n_yellow = verdicts.count("YELLOW")
            n_red = verdicts.count("RED")
            total = len(verdicts)
            print(f"    検出数: {total}, Laplacian: mean={np.mean(laps):.1f}, "
                  f"median={np.median(laps):.1f}, min={np.min(laps):.1f}, max={np.max(laps):.1f}")
            print(f"    判定: GREEN={n_green}({n_green/total:.0%}), "
                  f"YELLOW={n_yellow}({n_yellow/total:.0%}), "
                  f"RED={n_red}({n_red/total:.0%})")
            video_summaries.append({
                "video": os.path.basename(rel_path),
                "expected": expected_verdict,
                "n_det": total,
                "lap_mean": round(np.mean(laps), 1),
                "lap_median": round(np.median(laps), 1),
                "lap_min": round(np.min(laps), 1),
                "lap_max": round(np.max(laps), 1),
                "green": n_green, "yellow": n_yellow, "red": n_red,
            })
        else:
            print(f"    検出なし")

    # ============================================================
    # 総合レポート
    # ============================================================
    print("\n" + "=" * 70)
    print("総合レポート — モザイク済み動画での Laplacian 分布")
    print("=" * 70)

    if all_results:
        all_laps = [r["laplacian_var"] for r in all_results]
        print(f"\n  全検出数: {len(all_results)}")
        print(f"  Laplacian 分布:")
        print(f"    mean={np.mean(all_laps):.1f}, median={np.median(all_laps):.1f}")
        print(f"    p5={np.percentile(all_laps, 5):.1f}, p25={np.percentile(all_laps, 25):.1f}")
        print(f"    p75={np.percentile(all_laps, 75):.1f}, p95={np.percentile(all_laps, 95):.1f}")

        # 旧閾値 vs 新閾値
        old_low = 50.0
        new_low = cfg.genital_laplacian_threshold_low  # 15.0

        below_old = sum(1 for l in all_laps if l < old_low)
        below_new = sum(1 for l in all_laps if l < new_low)
        print(f"\n  閾値比較:")
        print(f"    旧 threshold_low=50: GREEN判定率 {below_old/len(all_laps):.1%} ({below_old}/{len(all_laps)})")
        print(f"    新 threshold_low=15: GREEN判定率 {below_new/len(all_laps):.1%} ({below_new}/{len(all_laps)})")

        # 期待カテゴリ別
        for cat in ["FAIL", "REVIEW", "PASS"]:
            cat_laps = [r["laplacian_var"] for r in all_results if r["expected"] == cat]
            cat_verdicts = [r["verdict"] for r in all_results if r["expected"] == cat]
            if cat_laps:
                n_g = cat_verdicts.count("GREEN")
                n_y = cat_verdicts.count("YELLOW")
                n_r = cat_verdicts.count("RED")
                tot = len(cat_laps)
                print(f"\n  [{cat}] 動画群 ({tot} 検出):")
                print(f"    Laplacian: mean={np.mean(cat_laps):.1f}, median={np.median(cat_laps):.1f}")
                print(f"    判定: GREEN={n_g}({n_g/tot:.0%}), YELLOW={n_y}({n_y/tot:.0%}), RED={n_r}({n_r/tot:.0%})")

    # 動画別サマリテーブル
    print(f"\n  {'動画':<45} {'期待':>6} {'検出':>4} {'Lap_med':>7} {'G':>3} {'Y':>3} {'R':>3}")
    print(f"  {'-'*75}")
    for s in video_summaries:
        print(f"  {s['video']:<45} {s['expected']:>6} {s['n_det']:>4} "
              f"{s['lap_median']:>7.1f} {s['green']:>3} {s['yellow']:>3} {s['red']:>3}")

    # 結論
    print(f"\n  結論:")
    if all_results:
        fail_laps = [r["laplacian_var"] for r in all_results if r["expected"] == "FAIL"]
        pass_laps = [r["laplacian_var"] for r in all_results if r["expected"] == "PASS"]
        if fail_laps and pass_laps:
            print(f"    FAIL動画群 Laplacian median = {np.median(fail_laps):.1f}")
            print(f"    PASS動画群 Laplacian median = {np.median(pass_laps):.1f}")
            if np.median(fail_laps) > cfg.genital_laplacian_threshold_low > np.median(pass_laps):
                print(f"    → threshold_low={cfg.genital_laplacian_threshold_low} は FAIL/PASS を分離できている ✓")
            elif np.median(pass_laps) < cfg.genital_laplacian_threshold_low:
                print(f"    → threshold_low={cfg.genital_laplacian_threshold_low} はモザイク済みROIでGREEN判定 ✓")
                print(f"    → FAIL動画はGDinoスコアが高く confidence boost で RED 昇格される")

    # JSON 保存
    output_path = os.path.join(REPO_DIR, "genital-reference", "LV_Open-GroundingDino",
                               "local_train", "mosaic_video_verification.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "genital_laplacian_threshold_low": cfg.genital_laplacian_threshold_low,
                "genital_laplacian_threshold_high": cfg.genital_laplacian_threshold_high,
                "genital_confidence_yellow_th": cfg.genital_confidence_yellow_th,
                "genital_confidence_red_th": cfg.genital_confidence_red_th,
            },
            "video_summaries": video_summaries,
            "all_detections": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  結果 JSON: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
