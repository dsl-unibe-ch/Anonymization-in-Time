"""Pure helpers for discovering videos and planning collision-free outputs.

This module deliberately depends only on the Python standard library so that
it can be imported and unit-tested without pulling in OpenCV, torch, or any of
the heavy ML pipeline. It is shared by the Video Processor GUI and the
``ait.process_videos`` CLI.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

# Supported video container extensions (compared case-insensitively).
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
)

# Characters that are safe across Windows/macOS/Linux directory names.
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_supported_video(path) -> bool:
    """Return True if ``path`` has a supported video extension (any case)."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def _relative_parts(path: Path, root) -> tuple:
    """Return ``path`` broken into parts relative to ``root`` when possible."""
    path = Path(path)
    if root is not None:
        try:
            return Path(path).relative_to(root).parts
        except ValueError:
            pass
    # Fall back to the path without its drive/anchor so keys stay comparable.
    parts = path.parts
    if path.anchor and parts and parts[0] == path.anchor:
        parts = parts[1:]
    return parts


def _sort_key(path: Path, root):
    """Deterministic ordering key based on the relative path, not basename.

    Case-folded parts drive the primary order; the original parts break ties so
    two paths differing only in case still sort deterministically.
    """
    parts = _relative_parts(path, root)
    return ([p.casefold() for p in parts], list(parts))


def discover_videos(root, recursive: bool = True) -> list:
    """Recursively discover supported videos under ``root``.

    - Recurses to arbitrary depth (unless ``recursive`` is False).
    - Matches extensions case-insensitively.
    - Ignores unsupported files and directories.
    - Returns each discovered path exactly once.
    - Sorts deterministically by path relative to ``root``.
    """
    root = Path(root)
    seen = set()
    found = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for candidate in iterator:
        if not candidate.is_file():
            continue
        if not is_supported_video(candidate):
            continue
        # De-duplicate on the resolved path to guard against symlink loops or
        # overlapping globs while still returning the original path object.
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        found.append(candidate)
    found.sort(key=lambda p: _sort_key(p, root))
    return found


def _sanitize(part: str) -> str:
    """Reduce a single path component to a filesystem-safe token."""
    cleaned = _UNSAFE_CHARS_RE.sub("_", part).strip("._")
    return cleaned or "_"


def _relative_output_name(path: Path, root) -> str:
    """Build an extension-free, filesystem-safe name from the relative path."""
    parts = list(_relative_parts(Path(path), root))
    if parts:
        parts[-1] = Path(parts[-1]).stem  # drop the extension from the file part
    tokens = [_sanitize(p) for p in parts if p not in ("", ".", "..")]
    return "__".join(t for t in tokens if t) or _sanitize(Path(path).stem)


def _short_digest(path: Path, root, salt: str = "") -> str:
    rel = "/".join(_relative_parts(Path(path), root)) or str(path)
    return hashlib.sha1((rel + "\0" + salt).encode("utf-8")).hexdigest()[:8]


def plan_output_names(video_paths, root=None) -> list:
    """Assign each video a collision-free output subdirectory name.

    Returns a list of ``(video_path, output_name)`` tuples in the same order as
    ``video_paths``. Unique stems keep their legacy ``<stem>`` folder name so
    existing layouts are preserved. Only videos whose stem collides with another
    video receive deterministic, relative-path-derived names, with a stable
    digest fallback when sanitization or case-folding would still collide.
    """
    paths = [Path(p) for p in video_paths]

    # Detect stem collisions case-insensitively: output directories may live on
    # a case-insensitive filesystem where ``Clip`` and ``clip`` are the same.
    stem_counts = Counter(p.stem.casefold() for p in paths)

    used = {}  # casefolded final name -> path it was assigned to
    assignments = [None] * len(paths)

    # First pass: assign legacy stem names to unique stems (priority).
    for idx, path in enumerate(paths):
        if stem_counts[path.stem.casefold()] == 1:
            name = path.stem
            used[name.casefold()] = path
            assignments[idx] = (path, name)

    # Second pass: derive names for colliding stems, keeping uniqueness.
    for idx, path in enumerate(paths):
        if assignments[idx] is not None:
            continue
        name = _relative_output_name(path, root)
        if name.casefold() in used:
            name = f"{name}_{_short_digest(path, root)}"
            salt = 1
            while name.casefold() in used:
                name = f"{_relative_output_name(path, root)}_{_short_digest(path, root, str(salt))}"
                salt += 1
        used[name.casefold()] = path
        assignments[idx] = (path, name)

    return assignments
