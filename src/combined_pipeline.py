"""顔＋性器 統合モザイク漏れ検知パイプライン

設計書 (0504_顔+性器AI検知.md) に基づく処理フロー:
  動画 → [顔パイプライン (SCRFD)] + [性器パイプライン (GroundingDino)] → 統合判定 → JSON

両パイプラインは同一フレーム抽出から独立して動作し、
最終的な統合判定は FAIL > REVIEW > PASS の優先順位で決定する。

Usage:
    python combined_pipeline.py --input video.mp4 [--output result.json]
    python combined_pipeline.py --input video.mp4 --output result.json \\
        --dino-checkpoint /path/to/checkpoint.pth
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

# ffmpegが見つからない場合に備えてPATHに追加
_CONDA_BIN = "/home/pan/miniforge3/bin"
if _CONDA_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _CONDA_BIN + ":" + os.environ.get("PATH", "")

import cv2
import numpy as np

# プロジェクト内パスを追加
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(REPO_DIR, "models")
VENDOR_DIR = os.path.join(REPO_DIR, "vendor")

sys.path.insert(0, SRC_DIR)
_gdino_lib = os.path.join(VENDOR_DIR, "grounding_dino")
if os.path.isdir(_gdino_lib) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from config import Config

# ============================================================
# 統合判定ロジック
# ============================================================

_VERDICT_RANK = {"PASS": 0, "REVIEW": 1, "FAIL": 2}


def merge_verdicts(face_verdict: str, genital_verdict: str) -> str:
    """
    2つのパイプラインの判定を統合する。
    FAIL > REVIEW > PASS の優先順位（最も深刻な判定を採用）。
    """
    rank_face    = _VERDICT_RANK.get(face_verdict,    0)
    rank_genital = _VERDICT_RANK.get(genital_verdict, 0)
    if rank_face >= rank_genital:
        return face_verdict
    return genital_verdict


# ============================================================
# フレーム抽出（共有）
# ============================================================

def get_video_duration_minutes(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (fc / fps) / 60.0 if fps > 0 else 0.0


def extract_frames(video_path: str, output_dir: str, cfg: Config) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    duration_min = get_video_duration_minutes(video_path)
    fps = cfg.fps_long if duration_min >= cfg.long_video_threshold_min else cfg.fps_normal
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale='min({cfg.max_resolution},iw)':'-1'",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%06d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return sorted(
        [os.path.join(output_dir, f)
         for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )


# ============================================================
# メインパイプライン
# ============================================================

def run_combined_pipeline(
    video_path: str,
    cfg: Config,
    dino_config: str,
    dino_checkpoint: str,
    output_path: Optional[str] = None,
    run_face: bool = True,
    run_genital: bool = True,
    box_threshold: float = 0.18,
    text_threshold: float = 0.15,
) -> dict:
    """
    顔＋性器 統合モザイク漏れ検知を実行する。

    Args:
        video_path       : 検査対象動画のパス
        cfg              : Config オブジェクト
        dino_config      : GroundingDino 設定ファイルのパス
        dino_checkpoint  : Fine-tuned GroundingDino チェックポイントのパス
        output_path      : 結果JSON の出力先（None の場合は出力しない）
        run_face         : 顔パイプラインを実行するか
        run_genital      : 性器パイプラインを実行するか
        box_threshold    : GroundingDino の box threshold
        text_threshold   : GroundingDino の text threshold

    Returns:
        統合判定結果の dict
    """
    start_time = time.time()
    video_path = os.path.abspath(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    face_result    = None
    genital_result = None

    # フレーム抽出は両パイプラインで共有する（一度だけ）
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_dir = os.path.join(tmp_dir, "frames")
        print(f"[1/N] フレーム抽出中...")
        frame_paths = extract_frames(video_path, frames_dir, cfg)
        print(f"      {len(frame_paths)} フレーム抽出完了")

        # --- 顔パイプライン ---
        if run_face:
            print("\n[顔パイプライン] SCRFD + ByteTrack + Laplacian/ACF/DCT")
            try:
                from pipeline import run_pipeline as run_face_pipeline
                face_result = run_face_pipeline(video_path, cfg)
                print(f"  → 顔判定: {face_result['verdict']}")
            except Exception as e:
                print(f"  [ERROR] 顔パイプライン失敗: {e}")
                face_result = {
                    "verdict": "ERROR",
                    "error": str(e),
                    "check_type": "mosaic_leak_detection_face",
                }

        # --- 性器パイプライン ---
        if run_genital:
            print("\n[性器パイプライン] Fine-tuned GroundingDino + Laplacian/ACF/DCT")
            try:
                from genital_pipeline import run_pipeline as run_genital_pipeline
                genital_result = run_genital_pipeline(
                    video_path,
                    cfg,
                    dino_config,
                    dino_checkpoint,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                print(f"  → 性器判定: {genital_result['verdict']}")
            except Exception as e:
                print(f"  [ERROR] 性器パイプライン失敗: {e}")
                genital_result = {
                    "verdict": "ERROR",
                    "error": str(e),
                    "check_type": "mosaic_leak_detection_genital",
                }

    # --- 統合判定 ---
    face_v    = (face_result    or {}).get("verdict", "PASS")
    genital_v = (genital_result or {}).get("verdict", "PASS")

    # ERROR は REVIEW として扱う（安全側に倒す）
    if face_v    == "ERROR": face_v    = "REVIEW"
    if genital_v == "ERROR": genital_v = "REVIEW"

    combined_verdict = merge_verdicts(face_v, genital_v)

    processing_time = round(time.time() - start_time, 1)

    combined_result = {
        "video_path"       : video_path,
        "check_type"       : "mosaic_leak_detection_combined",
        "verdict"          : combined_verdict,
        "face_verdict"     : face_v,
        "genital_verdict"  : genital_v,
        "processing_time_sec": processing_time,
        "pipelines": {
            "face"   : face_result,
            "genital": genital_result,
        },
    }

    # --- JSON 出力 ---
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, ensure_ascii=False, indent=2)
        print(f"\n結果保存: {output_path}")

    return combined_result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="顔＋性器 統合モザイク漏れ検知パイプライン"
    )
    parser.add_argument("--input",  required=True,  help="検査対象動画のパス")
    parser.add_argument("--output", default=None,   help="結果JSONの出力先")
    parser.add_argument(
        "--dino-checkpoint",
        default=os.path.join(MODELS_DIR, "gdino", "v3ft_best.pth"),
        help="Fine-tuned GroundingDino チェックポイントのパス",
    )
    parser.add_argument(
        "--dino-config",
        default=os.path.join(MODELS_DIR, "gdino", "cfg_odvg.py"),
        help="GroundingDino config ファイルのパス",
    )
    parser.add_argument(
        "--face-only",    action="store_true", help="顔パイプラインのみ実行"
    )
    parser.add_argument(
        "--genital-only", action="store_true", help="性器パイプラインのみ実行"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.18,
        help="GroundingDino の box threshold",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.15,
        help="GroundingDino の text threshold",
    )
    args = parser.parse_args()

    cfg = Config()
    run_face    = not args.genital_only
    run_genital = not args.face_only

    result = run_combined_pipeline(
        video_path       = args.input,
        cfg              = cfg,
        dino_config      = args.dino_config,
        dino_checkpoint  = args.dino_checkpoint,
        output_path      = args.output,
        run_face         = run_face,
        run_genital      = run_genital,
        box_threshold    = args.box_threshold,
        text_threshold   = args.text_threshold,
    )

    print(f"\n統合判定: {result['verdict']}")
    print(f"  顔    : {result['face_verdict']}")
    print(f"  性器  : {result['genital_verdict']}")

    # 終了コード
    exit_codes = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
    sys.exit(exit_codes.get(result["verdict"], 5))


if __name__ == "__main__":
    main()
