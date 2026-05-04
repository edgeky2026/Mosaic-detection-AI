"""モザイク漏れ検知 AI — メインパイプライン

設計書(0501)に基づく処理フロー:
  動画 → フレーム抽出 → SCRFD顔検出 → ByteTrack追跡 → モザイク解析 → 判定 → JSON出力

Usage:
    python pipeline.py --input video.mp4 [--output result.json] [--conf-th 0.3]
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
VENDOR_DIR = os.path.join(REPO_DIR, "vendor")
sys.path.insert(0, SRC_DIR)

# 既存資産のパスを追加
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from config import Config
from mosaic_analyzer import analyze_roi
from scorer import (
    FaceResult,
    FrameResult,
    FrameVerdict,
    VideoVerdict,
    aggregate_video_verdict,
    determine_face_verdict,
    determine_frame_verdict,
    extract_flagged_segments,
)


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

    # フレームファイルをソート済みで返す
    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )
    return frames


def load_detector(cfg: Config):
    """SCRFD-2.5g 検出器を初期化"""
    from scrfd.pub import SCRFD
    from scrfd.schemas import Threshold

    detector = SCRFD.from_path(cfg.scrfd_model_path, providers=list(cfg.scrfd_providers))
    threshold = Threshold(probability=cfg.scrfd_conf_th, nms=cfg.scrfd_nms_th)
    return detector, threshold


def load_detector_34g(cfg: Config):
    """施策D: SCRFD-34g 検出器を初期化"""
    from scrfd.pub import SCRFD
    from scrfd.schemas import Threshold

    detector = SCRFD.from_path(cfg.scrfd_34g_model_path, providers=list(cfg.scrfd_providers))
    threshold = Threshold(probability=cfg.scrfd_conf_th, nms=cfg.scrfd_nms_th)
    return detector, threshold


def detect_faces_scrfd(detector, threshold, bgr_image: np.ndarray, cfg: Config) -> np.ndarray:
    """SCRFD顔検出 → Nx5 [x1, y1, x2, y2, score]"""
    from PIL import Image

    rgb = Image.fromarray(bgr_image[:, :, ::-1])
    faces = detector.detect(rgb, threshold=threshold)

    dets = []
    for f in faces:
        x1 = float(f.bbox.upper_left.x)
        y1 = float(f.bbox.upper_left.y)
        x2 = float(f.bbox.lower_right.x)
        y2 = float(f.bbox.lower_right.y)
        score = float(f.probability)
        dets.append([x1, y1, x2, y2, score])

    if not dets:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(dets, dtype=np.float32)


def detect_faces_hybrid(
    detector_25g, threshold_25g,
    detector_34g, threshold_34g,
    bgr_image: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    施策D: SCRFD-34g ハイブリッド検出。
    まず 34g で検出し、2 顔以上のフレームでは 2.5g にフォールバックする。
    単顔フレームでは 34g が高精度であることが確認されている（総括文書§1.1）。
    複数顔フレームでは 34g は 2 人目を見逃すため 2.5g を使用する。
    """
    dets_34g = detect_faces_scrfd(detector_34g, threshold_34g, bgr_image, cfg)
    if len(dets_34g) <= 1:
        return dets_34g  # 単顔または未検出: 34g が高精度
    else:
        return detect_faces_scrfd(detector_25g, threshold_25g, bgr_image, cfg)  # 複数顔: 2.5g


def load_tracker(cfg: Config):
    """ByteTrackトラッカーを初期化"""
    from dataclasses import dataclass

    @dataclass
    class TrackerArgs:
        track_thresh: float = cfg.track_thresh
        track_buffer: int = cfg.track_buffer
        match_thresh: float = cfg.match_thresh
        mot20: bool = False

    from yolox.tracker.byte_tracker import BYTETracker
    tracker = BYTETracker(TrackerArgs(), frame_rate=cfg.tracker_frame_rate)
    return tracker


def extract_roi(frame: np.ndarray, bbox: Tuple[float, ...], padding: int) -> np.ndarray:
    """顔bboxからROIを切り出す（パディング付き）"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        # 空のROIの場合は小さな黒画像を返す
        roi = np.zeros((padding * 2, padding * 2, 3), dtype=np.uint8)
    return roi


def run_pipeline(video_path: str, cfg: Config, output_path: Optional[str] = None) -> dict:
    """
    メインパイプライン: 動画 → 判定結果JSON

    Returns:
        判定結果のdict
    """
    start_time = time.time()
    video_path = os.path.abspath(video_path)

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

        # fps計算（タイムスタンプ算出用）
        duration_min = get_video_duration_minutes(video_path)
        fps = cfg.fps_long if duration_min >= cfg.long_video_threshold_min else cfg.fps_normal
        sec_per_frame = 1.0 / fps

        # --- [2] モデル初期化 ---
        print(f"[2/5] モデル初期化中...")

        detector_25g, threshold_25g = load_detector(cfg)

        # 施策D: 34g ハイブリッド用に 34g 検出器を追加ロード
        detector_34g, threshold_34g = None, None
        if cfg.use_scrfd34g_hybrid:
            detector_34g, threshold_34g = load_detector_34g(cfg)
            print("      施策D: SCRFD-34g ハイブリッド有効")

        # 施策C: SegFormer 二次確認レイヤー
        segformer_checker = None
        if cfg.use_segformer_confirmation:
            from segformer_checker import SegFormerChecker
            device = "cuda" if "CUDA" in cfg.scrfd_providers[0] else "cpu"
            segformer_checker = SegFormerChecker(
                model_path=cfg.segformer_model_path,
                face_classes=cfg.segformer_face_classes,
                device=device,
            )
            print("      施策C: SegFormer二次確認レイヤー有効")

        tracker = load_tracker(cfg)

        # トラック履歴（track_id → 出現フレーム数）
        track_history: Dict[int, int] = {}
        frame_results: List[FrameResult] = []

        # --- [3] フレームループ ---
        print(f"[3/5] 顔検出・追跡・解析中...")
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            timestamp_sec = frame_idx * sec_per_frame

            # 顔検出（施策D: ハイブリッド or 通常）
            if cfg.use_scrfd34g_hybrid and detector_34g is not None:
                dets = detect_faces_hybrid(
                    detector_25g, threshold_25g,
                    detector_34g, threshold_34g,
                    frame, cfg,
                )
            else:
                dets = detect_faces_scrfd(detector_25g, threshold_25g, frame, cfg)

            # ByteTrack追跡
            img_info = (h, w, 1.0)
            img_size = (h, w)
            tracker.det_thresh = cfg.track_thresh
            online_targets = tracker.update(dets, img_info, img_size)

            # トラック履歴更新
            for t in online_targets:
                tid = t.track_id
                track_history[tid] = track_history.get(tid, 0) + 1

            # --- モザイク解析 ---
            face_results: List[FaceResult] = []

            for t in online_targets:
                tid = t.track_id
                tlbr = t.tlbr  # [x1, y1, x2, y2]

                # 偽陽性棄却: 追跡フレーム数が不足（単発ノイズ）
                if track_history.get(tid, 0) < cfg.min_track_frames:
                    continue

                # ROI切り出し
                roi = extract_roi(frame, tlbr, cfg.roi_padding)

                # 3指標解析
                analysis = analyze_roi(roi, cfg.roi_normalize_size)

                # 判定（施策B: confidence を渡す）
                verdict = determine_face_verdict(
                    laplacian_var=analysis["laplacian_var"],
                    block_size=analysis["block_size"],
                    periodicity=analysis["periodicity"],
                    cfg=cfg,
                    confidence=float(t.score),  # 施策B
                )

                # 施策C: SegFormer 二次確認
                if segformer_checker is not None and verdict != FrameVerdict.GREEN:
                    face_ratio = segformer_checker.face_pixel_ratio(roi)
                    if face_ratio < cfg.segformer_face_ratio_min:
                        # SegFormer が顔を検出しない → SCRFD 誤検出の可能性
                        # GREEN に降格して偽陽性を抑制
                        verdict = FrameVerdict.GREEN

                face_results.append(FaceResult(
                    face_id=tid,
                    bbox=tuple(tlbr.tolist()),
                    confidence=float(t.score),
                    laplacian_var=analysis["laplacian_var"],
                    block_size=analysis["block_size"],
                    periodicity=analysis["periodicity"],
                    verdict=verdict,
                ))

            # フレーム判定
            frame_verdict = determine_frame_verdict(face_results)
            frame_results.append(FrameResult(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                faces=face_results,
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
                    "face_id": seg.face_id,
                    "avg_laplacian_var": round(seg.avg_laplacian_var, 1),
                }
                for seg in flagged_segments
            ],
            "metadata": {
                "model_versions": {
                    "scrfd": "2.5g",
                    "analyzer": "v1.0",
                },
                "thresholds": {
                    "laplacian_low": cfg.laplacian_threshold_low,
                    "laplacian_high": cfg.laplacian_threshold_high,
                    "block_size_green": cfg.block_size_green,
                    "block_size_yellow": cfg.block_size_yellow,
                    "periodicity_threshold": cfg.periodicity_threshold,
                    "scrfd_conf_th": cfg.scrfd_conf_th,
                },
                "fps_used": fps,
                "duration_min": round(duration_min, 1),
            },
        }

        # JSON出力
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[5/5] 結果出力: {output_path}")
        else:
            print(f"[5/5] 完了")

        # サマリー表示
        _print_summary(result)

        return result


def _compute_max_consecutive_from_results(frame_results: List[FrameResult]) -> int:
    max_run = 0
    current_run = 0
    for fr in frame_results:
        if fr.verdict == FrameVerdict.RED:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _print_summary(result: dict):
    """結果サマリーをコンソール表示"""
    verdict = result["verdict"]
    s = result["summary"]

    # 色付き表示
    color_map = {"PASS": "\033[92m", "REVIEW": "\033[93m", "FAIL": "\033[91m"}
    reset = "\033[0m"
    color = color_map.get(verdict, "")

    print(f"\n{'='*60}")
    print(f"  判定結果: {color}{verdict}{reset}")
    print(f"  処理時間: {s['processing_time_sec']}秒")
    print(f"  フレーム: GREEN={s['green_frames']} / YELLOW={s['yellow_frames']} / RED={s['red_frames']} (全{s['total_frames']})")
    print(f"  RED率: {s['red_rate']*100:.1f}%  連続RED最大: {s['max_consecutive_red']}")

    if result["flagged_segments"]:
        print(f"\n  警告区間:")
        for seg in result["flagged_segments"][:10]:
            start = _format_time(seg["start_sec"])
            end = _format_time(seg["end_sec"])
            print(f"    {seg['severity']:6s} {start}〜{end} (顔ID:{seg['face_id']}, Laplacian:{seg['avg_laplacian_var']:.0f})")

    print(f"{'='*60}\n")


def _format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="モザイク漏れ検知 AI パイプライン")
    parser.add_argument("--input", "-i", required=True, help="入力動画パス")
    parser.add_argument("--output", "-o", default=None, help="出力JSONパス（省略時はコンソールのみ）")
    parser.add_argument("--conf-th", type=float, default=None, help="SCRFD信頼度閾値（default: 0.3）")
    parser.add_argument("--laplacian-low", type=float, default=None, help="Laplacian低閾値")
    parser.add_argument("--laplacian-high", type=float, default=None, help="Laplacian高閾値")
    parser.add_argument("--cpu", action="store_true", help="CPUのみで実行")
    args = parser.parse_args()

    cfg = Config()

    if args.conf_th is not None:
        cfg.scrfd_conf_th = args.conf_th
    if args.laplacian_low is not None:
        cfg.laplacian_threshold_low = args.laplacian_low
    if args.laplacian_high is not None:
        cfg.laplacian_threshold_high = args.laplacian_high
    if args.cpu:
        cfg.scrfd_providers = ("CPUExecutionProvider",)

    result = run_pipeline(args.input, cfg, args.output)

    # 終了コード: FAIL=2, REVIEW=1, PASS=0
    exit_code = {"PASS": 0, "REVIEW": 1, "FAIL": 2}.get(result["verdict"], 0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
