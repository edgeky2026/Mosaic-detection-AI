"""モザイク漏れ検知 AI — スコアリング・集約モジュール

フレーム単位の3指標からGREEN/YELLOW/REDを判定し、
動画全体としてPASS/REVIEW/FAILに集約する。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from config import Config


class FrameVerdict(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class VideoVerdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


@dataclass
class FaceResult:
    """1顔ROIの解析結果"""
    face_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    laplacian_var: float
    block_size: Optional[int]
    periodicity: float
    verdict: FrameVerdict


@dataclass
class FrameResult:
    """1フレームの判定結果"""
    frame_idx: int
    timestamp_sec: float
    faces: List[FaceResult]
    verdict: FrameVerdict  # フレーム内で最も深刻な判定


@dataclass
class FlaggedSegment:
    """警告区間"""
    start_sec: float
    end_sec: float
    severity: FrameVerdict
    face_id: int
    avg_laplacian_var: float


def determine_face_verdict(
    laplacian_var: float,
    block_size: Optional[int],
    periodicity: float,
    cfg: Config,
    confidence: float = 0.0,  # 施策B: SCRFD confidence score
) -> FrameVerdict:
    """
    3指標（+施策B: SCRFD confidence）から1顔の判定を返す。

    判定ロジック:
      GREEN: Laplacian低 AND (ブロック大 OR 周期性高)
      YELLOW: Laplacian中 AND (ブロック中 OR 周期性中)
      RED: 上記いずれにも該当しない

    施策B: use_confidence_boost=True の場合、高信頼度検出は自動昇格
      confidence >= confidence_red_th  → 最低でもRED
      confidence >= confidence_yellow_th → 最低でもYELLOW
    """
    bs = block_size if block_size is not None else 0

    # GREEN（モザイク十分）
    if laplacian_var < cfg.laplacian_threshold_low:
        verdict = FrameVerdict.GREEN
    # 粗いブロックモザイクの補正（3指標設計の核心）
    # Laplacian高値でも、ブロックサイズ大 + DCT周期性高 → GREEN
    elif bs >= cfg.block_size_green and periodicity > cfg.periodicity_threshold:
        verdict = FrameVerdict.GREEN
    # YELLOW（境界的）
    elif laplacian_var < cfg.laplacian_threshold_high:
        if bs >= cfg.block_size_yellow or periodicity > cfg.periodicity_threshold_half:
            verdict = FrameVerdict.YELLOW
        else:
            verdict = FrameVerdict.RED
    # ブロックサイズ中 + 周期性中 → YELLOW
    elif bs >= cfg.block_size_yellow and periodicity > cfg.periodicity_threshold_half:
        verdict = FrameVerdict.YELLOW
    else:
        # RED（モザイク不足）
        verdict = FrameVerdict.RED

    # =========================================================
    # 施策B: SCRFD confidence による深刻度エスカレーション
    # 高信頼度で顔が見えている場合、指標解析が低値でも昇格させる
    # =========================================================
    if cfg.use_confidence_boost:
        severity = {FrameVerdict.GREEN: 0, FrameVerdict.YELLOW: 1, FrameVerdict.RED: 2}
        if confidence >= cfg.confidence_red_th:
            # 非常に高信頼度 → RED以上
            if severity[verdict] < severity[FrameVerdict.RED]:
                verdict = FrameVerdict.RED
        elif confidence >= cfg.confidence_yellow_th:
            # 高信頼度 → YELLOW以上
            if severity[verdict] < severity[FrameVerdict.YELLOW]:
                verdict = FrameVerdict.YELLOW

    return verdict


def determine_genital_verdict(
    laplacian_var: float,
    block_size: Optional[int],
    periodicity: float,
    cfg: Config,
    confidence: float = 0.0,
) -> FrameVerdict:
    """
    性器ROI の 3指標 + GDino confidence から判定を返す。

    顔用 determine_face_verdict との違い:
      - genital_laplacian_threshold_low/high を使用（原画 p5=6.9 と低いため）
      - GDino confidence boost の閾値が低い（GDino スコア帯 ≈ 0.20-0.65）

    判定ロジック: determine_face_verdict と同じ構造
    """
    bs = block_size if block_size is not None else 0

    lap_low = cfg.genital_laplacian_threshold_low
    lap_high = cfg.genital_laplacian_threshold_high

    if laplacian_var < lap_low:
        verdict = FrameVerdict.GREEN
    elif bs >= cfg.block_size_green and periodicity > cfg.periodicity_threshold:
        verdict = FrameVerdict.GREEN
    elif laplacian_var < lap_high:
        if bs >= cfg.block_size_yellow or periodicity > cfg.periodicity_threshold_half:
            verdict = FrameVerdict.YELLOW
        else:
            verdict = FrameVerdict.RED
    elif bs >= cfg.block_size_yellow and periodicity > cfg.periodicity_threshold_half:
        verdict = FrameVerdict.YELLOW
    else:
        verdict = FrameVerdict.RED

    # GDino confidence boost（施策B 相当）
    if cfg.use_confidence_boost:
        severity = {FrameVerdict.GREEN: 0, FrameVerdict.YELLOW: 1, FrameVerdict.RED: 2}
        if confidence >= cfg.genital_confidence_red_th:
            if severity[verdict] < severity[FrameVerdict.RED]:
                verdict = FrameVerdict.RED
        elif confidence >= cfg.genital_confidence_yellow_th:
            if severity[verdict] < severity[FrameVerdict.YELLOW]:
                verdict = FrameVerdict.YELLOW

    return verdict


def determine_frame_verdict(faces: List[FaceResult]) -> FrameVerdict:
    """フレーム内で最も深刻な判定を採用"""
    if not faces:
        return FrameVerdict.GREEN

    severity_order = {FrameVerdict.RED: 2, FrameVerdict.YELLOW: 1, FrameVerdict.GREEN: 0}
    worst = max(faces, key=lambda f: severity_order[f.verdict])
    return worst.verdict


def aggregate_video_verdict(
    frame_results: List[FrameResult],
    cfg: Config,
) -> VideoVerdict:
    """
    フレーム単位の結果を動画レベルに集約する。

    FAIL: RED率5%超 or 連続RED3フレーム以上
    REVIEW: REDが1件以上 or YELLOW率15%超
    PASS: それ以外
    """
    if not frame_results:
        return VideoVerdict.PASS

    total = len(frame_results)
    red_frames = [f for f in frame_results if f.verdict == FrameVerdict.RED]
    yellow_frames = [f for f in frame_results if f.verdict == FrameVerdict.YELLOW]

    red_rate = len(red_frames) / total
    yellow_rate = len(yellow_frames) / total

    # 連続REDフレーム区間の最大長を算出
    max_consecutive_red = _compute_max_consecutive(frame_results, FrameVerdict.RED)

    # FAIL
    if red_rate > cfg.red_rate_fail or max_consecutive_red >= cfg.consecutive_red_fail:
        return VideoVerdict.FAIL

    # REVIEW
    if len(red_frames) > 0 or yellow_rate > cfg.yellow_rate_review:
        return VideoVerdict.REVIEW

    # PASS
    return VideoVerdict.PASS


def extract_flagged_segments(
    frame_results: List[FrameResult],
) -> List[FlaggedSegment]:
    """連続するRED/YELLOWフレームを区間にまとめる"""
    segments: List[FlaggedSegment] = []
    if not frame_results:
        return segments

    current_severity = None
    start_sec = 0.0
    face_id = 0
    laplacian_sum = 0.0
    count = 0

    for fr in frame_results:
        if fr.verdict in (FrameVerdict.RED, FrameVerdict.YELLOW):
            if current_severity is None:
                current_severity = fr.verdict
                start_sec = fr.timestamp_sec
                count = 0
                laplacian_sum = 0.0
            elif fr.verdict != current_severity:
                # 区間の切れ目
                segments.append(FlaggedSegment(
                    start_sec=start_sec,
                    end_sec=fr.timestamp_sec,
                    severity=current_severity,
                    face_id=face_id,
                    avg_laplacian_var=laplacian_sum / max(count, 1),
                ))
                current_severity = fr.verdict
                start_sec = fr.timestamp_sec
                count = 0
                laplacian_sum = 0.0

            # 最も深刻な顔のLaplacian値を加算
            worst_face = max(fr.faces, key=lambda f: f.laplacian_var) if fr.faces else None
            if worst_face:
                laplacian_sum += worst_face.laplacian_var
                face_id = worst_face.face_id
            count += 1
        else:
            if current_severity is not None:
                segments.append(FlaggedSegment(
                    start_sec=start_sec,
                    end_sec=fr.timestamp_sec,
                    severity=current_severity,
                    face_id=face_id,
                    avg_laplacian_var=laplacian_sum / max(count, 1),
                ))
                current_severity = None

    # 最後の区間
    if current_severity is not None:
        segments.append(FlaggedSegment(
            start_sec=start_sec,
            end_sec=frame_results[-1].timestamp_sec + 1.0,
            severity=current_severity,
            face_id=face_id,
            avg_laplacian_var=laplacian_sum / max(count, 1),
        ))

    return segments


def _compute_max_consecutive(frame_results: List[FrameResult], target: FrameVerdict) -> int:
    """指定verdictの最大連続フレーム数"""
    max_run = 0
    current_run = 0
    for fr in frame_results:
        if fr.verdict == target:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run
