import sys
import os
from pathlib import Path
import re

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    if relative_path is None:
        raise ValueError("resource_path() expected a non-None path")
    if not isinstance(relative_path, (str, os.PathLike)):
        raise TypeError(
            f"resource_path() expected str or os.PathLike, got {type(relative_path).__name__}"
        )

    p = Path(relative_path)
    if p.is_absolute():
        return str(p)

    # PyInstaller bundled runtime path
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return str(Path(bundle_root) / p)

    # dev environment: resolve relative to repo root (folder containing utils/)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / p)


def load_prompt(filename: str) -> str:

    if filename is None:
        raise ValueError("load_prompt() expected a non-None filename")

    include_pattern = re.compile(r"@@INCLUDE:\s*(.+?)\s*@@")

    def _load_inner(name: str, depth: int, seen: set[str]) -> str:
        if depth > 10:
            raise ValueError(f"Prompt include depth exceeded for: {name}")
        if name in seen:
            raise ValueError(f"Circular @@INCLUDE detected for: {name}")

        real_path = resource_path(name)
        prompt_path = Path(real_path)

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        text = prompt_path.read_text(encoding="utf-8")

        def _replace(match: re.Match) -> str:
            included = match.group(1).strip()
            return _load_inner(included, depth + 1, seen | {name})

        return include_pattern.sub(_replace, text)

    return _load_inner(filename, 0, set())