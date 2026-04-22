from __future__ import annotations

from pathlib import Path


CHECKPOINT_PATTERNS = {
    "0+4": [
        "checkpoints/0+0+4/*.pth",
        "outputs/ckpt/0+0+4/*.pth",
    ],
    "0+4_wobound": [
        "checkpoints/0+0+4_wobound/*.pth",
        "outputs/ckpt/0+0+4 wobound/*.pth",
    ],
    "4+4": [
        "checkpoints/0+4+4/*.pth",
        "outputs/ckpt/0+4+4_32bps/*.pth",
    ],
}


def _latest_match(patterns: list[str]) -> str | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(latest.resolve())


def resolve_checkpoint(ckpt_name: str, ckpt_path: str | bool | None = None) -> str:
    if ckpt_path and str(ckpt_path).lower() != "false":
        return str(Path(str(ckpt_path)).expanduser().resolve())

    patterns = CHECKPOINT_PATTERNS.get(ckpt_name)
    if not patterns:
        available = ", ".join(sorted(CHECKPOINT_PATTERNS))
        raise ValueError(f"Unknown ckpt_name '{ckpt_name}'. Available values: {available}")

    resolved = _latest_match(patterns)
    if resolved:
        return resolved

    pattern_text = ", ".join(patterns)
    raise FileNotFoundError(
        "Could not resolve a checkpoint automatically. "
        f"Looked under: {pattern_text}. Pass --ckpt_path or ckpt_path=... explicitly."
    )
