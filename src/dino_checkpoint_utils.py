"""GroundingDINO checkpoint resolution helpers."""

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
GSAM2_DIR = REPO_DIR / "genital-reference" / "LV_Grounded-SAM-2"
LOCAL_TRAIN_DIR = REPO_DIR / "genital-reference" / "LV_Open-GroundingDino" / "local_train"

LEGACY_GDRIVE_BEST = GSAM2_DIR / "gdino_checkpoints" / "dino_gdrive_ft_best.pth"
LEGACY_GCP_FALLBACK = GSAM2_DIR / "gdino_checkpoints" / "dino_260430_checkpoint0002.pth"


def _latest_match(pattern: str) -> str | None:
    matches = sorted(
        (path for path in LOCAL_TRAIN_DIR.glob(pattern) if path.is_file()),
        reverse=True,
    )
    if not matches:
        return None
    return str(matches[0])


def get_default_dino_checkpoint() -> str:
    candidates = [
        _latest_match("output_gdrive_ft_v3_*/checkpoint_best_regular.pth"),
        _latest_match("output_gdrive_ft_v3_*/checkpoint000*.pth"),
        str(LEGACY_GDRIVE_BEST),
        str(LEGACY_GCP_FALLBACK),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return str(LEGACY_GCP_FALLBACK)


def resolve_dino_checkpoint(checkpoint_path: str | None) -> str:
    checked = []
    seen = set()
    candidates = []

    if checkpoint_path:
        candidates.append(str(Path(checkpoint_path).expanduser()))

    candidates.extend(
        [
            get_default_dino_checkpoint(),
            str(LEGACY_GDRIVE_BEST),
            str(LEGACY_GCP_FALLBACK),
        ]
    )

    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if Path(candidate).is_file():
            return candidate

    raise FileNotFoundError(
        "GroundingDino checkpoint not found. Checked: " + ", ".join(checked)
    )