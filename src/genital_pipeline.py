"""性器モザイク漏れ検知 AI — メインパイプライン

設計書(0503_性器実装報告書.md)に基づく処理フロー:
  動画 → フレーム抽出 → Fine-tuned GroundingDino 性器検出 → モザイク解析 → 判定 → JSON出力

顔パイプライン(pipeline.py)との主な違い:
  - 検出モデル: SCRFD → Fine-tuned GroundingDino ("vagina . penis .")
  - トラッキング: ByteTrack なし（フレーム独立処理）
  - check_type: "mosaic_leak_detection_genital"

Usage:
    python genital_pipeline.py --input video.mp4 [--output result.json]
    python genital_pipeline.py --input video.mp4 \\
        --dino-checkpoint /path/to/checkpoint0007.pth \\
        --dino-config /path/to/cfg_odvg.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# GroundingDino ライブラリのパスを追加
_gdino_lib = os.path.join(VENDOR_DIR, "grounding_dino")
if os.path.isdir(_gdino_lib) and _gdino_lib not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from config import Config
from dino_checkpoint_utils import get_default_dino_checkpoint, resolve_dino_checkpoint
from mosaic_analyzer import analyze_roi
from scorer import (
    FaceResult,
    FrameResult,
    FrameVerdict,
    VideoVerdict,
    aggregate_video_verdict,
    determine_genital_verdict,
    determine_frame_verdict,
    extract_flagged_segments,
)

# --- デフォルトチェックポイントパス ---
# 既定では最新の v3FT best を優先し、存在しない場合のみ旧チェックポイントへフォールバックする。
DEFAULT_DINO_CHECKPOINT = get_default_dino_checkpoint()
DEFAULT_DINO_CONFIG = os.path.join(MODELS_DIR, "gdino", "cfg_odvg.py")
GENITAL_TEXT_PROMPT = "vagina . penis . anus ."
BOX_THRESHOLD = 0.18   # 施策Hモデル全体最適値（IoU=0.493, Coverage=0.647, hukusu回復）
TEXT_THRESHOLD = 0.15  # FTモデルでは無効だが互換性のため維持

# 性器ROIの最小サイズ（フレーム面積に対する比率）
# 0.001 = フレーム面積の 0.1% 以上の bbox のみ対象
MIN_BOX_AREA_RATIO = 0.001
MAX_BOX_AREA_RATIO = 0.30   # フレーム面積の 30% 超は体部位の巨大誤検出として棄却
MAX_ASPECT_RATIO = 4.0      # 施策J: max(w/h, h/w) > 4.0 の極端に細長い検出を棄却

def get_video_duration_minutes(video_path: str) -> float:
    """動画の長さを分で返す"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return (frame_count / fps) / 60.0


def extract_frames(video_path: str, output_dir: str, cfg: Config) -> List[str]:
    """ffmpegでフレームを抽出し、ファイルパスのリストを返す"""
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

    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )
    return frames


def load_grounding_dino(dino_config: str, dino_checkpoint: str):
    """Fine-tuned GroundingDino モデルをロード"""
    try:
        from grounding_dino.groundingdino.util.inference import load_model
    except ImportError as e:
        raise ImportError(
            "GroundingDino が見つかりません。以下を実行してください:\n"
            f"  cd {GSAM2_DIR}\n"
            "  pip install --no-build-isolation -e grounding_dino\n"
            f"元のエラー: {e}"
        )

    if not os.path.isfile(dino_config):
        raise FileNotFoundError(f"GroundingDino config not found: {dino_config}")
    if not os.path.isfile(dino_checkpoint):
        raise FileNotFoundError(
            f"GroundingDino checkpoint not found: {dino_checkpoint}\n"
            "  --dino-checkpoint フラグで利用可能な checkpoint のパスを指定してください。"
        )

    device = "cuda" if _cuda_available() else "cpu"
    model = load_model(dino_config, dino_checkpoint, device=device)
    return model, device


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def detect_genitals(
    model,
    device: str,
    bgr_image: np.ndarray,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
) -> List[Dict]:
    """
    GroundingDino で性器を検出。

    Returns:
        List of dicts: [{"bbox": [x1,y1,x2,y2], "score": float, "label": str}, ...]
        座標は元画像のピクセル座標（xyxy形式）
    """
    from PIL import Image as PILImage
    try:
        import torch
        from grounding_dino.groundingdino.util.inference import predict, load_image
        from torchvision.ops import box_convert
    except ImportError as e:
        raise ImportError(f"GroundingDino inference モジュールが見つかりません: {e}")

    h, w = bgr_image.shape[:2]
    rgb_pil = PILImage.fromarray(bgr_image[:, :, ::-1])

    # GroundingDino が期待する前処理
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(rgb_pil)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=GENITAL_TEXT_PROMPT,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    if len(boxes) == 0:
        return []

    # boxes は [cx, cy, w, h] の正規化座標 → xyxy ピクセル座標に変換
    boxes_xyxy = box_convert(
        boxes=boxes * torch.tensor([w, h, w, h], dtype=torch.float32),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).numpy()

    results = []
    for i, (box, score, label) in enumerate(zip(boxes_xyxy, logits.numpy(), phrases)):
        x1, y1, x2, y2 = box.tolist()
        # クリッピング
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(w), x2)
        y2 = min(float(h), y2)

        # サイズフィルタ（フレーム面積の 0.1% 未満は誤検出として棄却）
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area < w * h * MIN_BOX_AREA_RATIO:
            continue
        # 最大サイズフィルタ（フレーム面積の 30% 超は体部位の巨大FPとして棄却）
        if box_area > w * h * MAX_BOX_AREA_RATIO:
            continue

        # 施策J: アスペクト比フィルタ（極端に細長いBBoxは体部位FPとして棄却）
        bw = x2 - x1
        bh = y2 - y1
        if bw > 0 and bh > 0:
            ar = max(bw / bh, bh / bw)
            if ar > MAX_ASPECT_RATIO:
                continue

        results.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "label": label.strip().lower(),
        })

    return results


def extract_roi(frame: np.ndarray, bbox: List[float], padding: int) -> np.ndarray:
    """bboxからROIを切り出す（パディング付き）"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        roi = np.zeros((padding * 2, padding * 2, 3), dtype=np.uint8)
    return roi


def _compute_max_consecutive_from_results(frame_results: List[FrameResult]) -> int:
    """連続RED フレームの最大長を算出"""
    max_consecutive = 0
    current = 0
    for fr in frame_results:
        if fr.verdict == FrameVerdict.RED:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0
    return max_consecutive


def run_pipeline(
    video_path: str,
    cfg: Config,
    dino_config: str,
    dino_checkpoint: str,
    box_threshold: float = BOX_THRESHOLD,
    text_threshold: float = TEXT_THRESHOLD,
    output_path: Optional[str] = None,
) -> dict:
    """
    性器モザイク漏れ検知メインパイプライン

    Args:
        video_path: 検査対象動画のパス
        cfg: Config オブジェクト（Laplacian 閾値等）
        dino_config: GroundingDino 設定ファイルのパス
        dino_checkpoint: Fine-tuned GroundingDino チェックポイントのパス
        output_path: 結果JSONの出力先（None の場合は出力しない）

    Returns:
        判定結果のdict（顔パイプラインと同フォーマット + check_type フィールド）
    """
    start_time = time.time()
    video_path = os.path.abspath(video_path)
    dino_checkpoint = resolve_dino_checkpoint(dino_checkpoint)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # --- [1] フレーム抽出 ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_dir = os.path.join(tmp_dir, "frames")
        print(f"[1/5] フレーム抽出中...")
        frame_paths = extract_frames(video_path, frames_dir, cfg)
        print(f"      {len(frame_paths)} フレーム抽出完了")

        if not frame_paths:
            raise RuntimeError("No frames extracted")

        duration_min = get_video_duration_minutes(video_path)
        fps = cfg.fps_long if duration_min >= cfg.long_video_threshold_min else cfg.fps_normal
        sec_per_frame = 1.0 / fps

        # --- [2] モデル初期化 ---
        print(f"[2/5] GroundingDino モデル初期化中...")
        model, device = load_grounding_dino(dino_config, dino_checkpoint)
        print(f"      device={device}")

        frame_results: List[FrameResult] = []

        # --- [3] フレームループ ---
        print(f"[3/5] 性器検出・解析中...")
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            timestamp_sec = frame_idx * sec_per_frame

            # 性器検出（GroundingDino）
            detections = detect_genitals(
                model,
                device,
                frame,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # --- モザイク解析 ---
            genital_results: List[FaceResult] = []

            for det_idx, det in enumerate(detections):
                bbox = det["bbox"]
                score = det["score"]
                label = det["label"]

                # ROI切り出し
                roi = extract_roi(frame, bbox, cfg.roi_padding)

                # 3指標解析（mosaic_analyzer.py 再利用）
                analysis = analyze_roi(roi, cfg.roi_normalize_size)

                # 判定（性器専用閾値で判定）
                # confidence として GroundingDino の検出スコアを渡す
                verdict = determine_genital_verdict(
                    laplacian_var=analysis["laplacian_var"],
                    block_size=analysis["block_size"],
                    periodicity=analysis["periodicity"],
                    cfg=cfg,
                    confidence=score,
                )

                genital_results.append(FaceResult(
                    face_id=det_idx,
                    bbox=tuple(bbox),
                    confidence=score,
                    laplacian_var=analysis["laplacian_var"],
                    block_size=analysis["block_size"],
                    periodicity=analysis["periodicity"],
                    verdict=verdict,
                ))

            # フレーム判定
            frame_verdict = determine_frame_verdict(genital_results)
            frame_results.append(FrameResult(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                faces=genital_results,
                verdict=frame_verdict,
            ))

        print(f"      {len(frame_results)} フレーム処理完了")

        # --- [4] 動画全体の判定 ---
        print(f"[4/5] 動画判定集約中...")
        video_verdict = aggregate_video_verdict(frame_results, cfg)

        # --- [5] 結果構築 ---
        print(f"[5/5] 結果構築中...")
        flagged_segments = extract_flagged_segments(frame_results)

        green_count = sum(1 for f in frame_results if f.verdict == FrameVerdict.GREEN)
        yellow_count = sum(1 for f in frame_results if f.verdict == FrameVerdict.YELLOW)
        red_count = sum(1 for f in frame_results if f.verdict == FrameVerdict.RED)

        processing_time = time.time() - start_time

        result = {
            "video_path": video_path,
            "check_type": "mosaic_leak_detection_genital",
            "verdict": video_verdict.value,
            "summary": {
                "total_frames": len(frame_results),
                "green_frames": green_count,
                "yellow_frames": yellow_count,
                "red_frames": red_count,
                "red_rate": red_count / max(len(frame_results), 1),
                "yellow_rate": yellow_count / max(len(frame_results), 1),
                "max_consecutive_red": _compute_max_consecutive_from_results(frame_results),
                "processing_time_sec": round(processing_time, 1),
            },
            "flagged_segments": [
                {
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "severity": seg.severity.value,
                    "region_id": seg.face_id,
                    "avg_laplacian_var": round(seg.avg_laplacian_var, 1),
                }
                for seg in flagged_segments
            ],
            "metadata": {
                "model_versions": {
                    "groundingdino": os.path.basename(dino_checkpoint),
                    "analyzer": "v1.0",
                },
                "thresholds": {
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "text_prompt": GENITAL_TEXT_PROMPT,
                    "laplacian_low": cfg.genital_laplacian_threshold_low,
                    "laplacian_high": cfg.genital_laplacian_threshold_high,
                    "block_size_green": cfg.block_size_green,
                    "block_size_yellow": cfg.block_size_yellow,
                },
                "fps_used": fps,
                "duration_min": round(duration_min, 1),
                "device": device,
            },
        }

        # JSON出力
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n結果保存: {output_path}")

        print(f"\n{'='*50}")
        print(f"  判定: {result['verdict']}")
        print(f"  GREEN: {green_count} / YELLOW: {yellow_count} / RED: {red_count}")
        print(f"  処理時間: {processing_time:.1f}秒")
        print(f"{'='*50}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="性器モザイク漏れ検知パイプライン (Fine-tuned GroundingDino)"
    )
    parser.add_argument("--input", required=True, help="検査対象動画のパス")
    parser.add_argument("--output", default=None, help="結果JSONの出力先（省略時は保存しない）")
    parser.add_argument(
        "--dino-checkpoint",
        default=get_default_dino_checkpoint(),
        help="GroundingDino チェックポイントパス（デフォルト: 最新 v3FT best を自動選択）",
    )
    parser.add_argument(
        "--dino-config",
        default=DEFAULT_DINO_CONFIG,
        help=f"GroundingDino 設定ファイルパス（デフォルト: {DEFAULT_DINO_CONFIG}）",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=BOX_THRESHOLD,
        help=f"GroundingDino 検出閾値（デフォルト: {BOX_THRESHOLD}）",
    )
    parser.add_argument(
        "--laplacian-low",
        type=float,
        default=None,
        help="Laplacian 低閾値（省略時は config.py のデフォルト値を使用）",
    )
    parser.add_argument(
        "--laplacian-high",
        type=float,
        default=None,
        help="Laplacian 高閾値（省略時は config.py のデフォルト値を使用）",
    )
    parser.add_argument("--cpu", action="store_true", help="CPU のみで実行（推奨しない: 非常に低速）")

    args = parser.parse_args()

    if args.cpu:
        # CPU 強制モードの場合は CUDA を無効化
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cfg = Config()
    if args.laplacian_low is not None:
        cfg.genital_laplacian_threshold_low = args.laplacian_low
    if args.laplacian_high is not None:
        cfg.genital_laplacian_threshold_high = args.laplacian_high

    try:
        result = run_pipeline(
            video_path=args.input,
            cfg=cfg,
            dino_config=args.dino_config,
            dino_checkpoint=args.dino_checkpoint,
            box_threshold=args.box_threshold,
            output_path=args.output,
        )

        # 終了コード: 0=PASS, 1=REVIEW, 2=FAIL
        verdict = result.get("verdict", "PASS")
        exit_code = {"PASS": 0, "REVIEW": 1, "FAIL": 2}.get(verdict, 0)
        sys.exit(exit_code)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(3)
    except ImportError as e:
        print(f"[ERROR] ライブラリが見つかりません:\n{e}", file=sys.stderr)
        sys.exit(4)
    except Exception as e:
        import traceback
        print(f"[ERROR] 予期しないエラー: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(5)


if __name__ == "__main__":
    main()
