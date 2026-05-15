"""GroundingDINO checkpoint resolution helpers.

GitHub版: models/gdino/ 配下の v3ft_best.pth を優先的に解決する。
"""

import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
MODELS_DIR = REPO_DIR / "models" / "gdino"

# 優先順位リスト（上から順に探索）
_CANDIDATES = [
    MODELS_DIR / "v3ft_best.pth",
    MODELS_DIR / "dino_local_ft_ep4_best.pth",
]


def get_default_dino_checkpoint() -> str:
    """利用可能な最良の GroundingDINO checkpoint パスを返す。"""
    for candidate in _CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    # 全候補が見つからない場合は v3ft_best.pth パスを返す（ロード時にエラーで報告される）
    return str(_CANDIDATES[0])


def resolve_dino_checkpoint(checkpoint_path: str | None) -> str:
    """明示パスがあればそれを優先、なければ get_default_dino_checkpoint() を返す。"""
    if checkpoint_path:
        expanded = str(Path(checkpoint_path).expanduser())
        if Path(expanded).is_file():
            return expanded
    return get_default_dino_checkpoint()
