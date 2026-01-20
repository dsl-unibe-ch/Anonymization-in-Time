"""
Device selection and cleanup helpers for torch.

Supports auto-detect across CUDA, Apple MPS, and CPU.
"""

from __future__ import annotations

def resolve_device(requested: str | None = "auto") -> str:
    """
    Resolve the best torch device based on user request and availability.

    Order:
    1) Respect explicit 'cuda' or 'mps' if available, otherwise fall back to CPU.
    2) For 'auto', prefer CUDA, then MPS, else CPU.
    """
    try:
        import torch
    except Exception:
        return "cpu"

    req = (requested or "auto").lower()

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"

    if req == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "mps":
        return "mps" if mps_available() else "cpu"

    return "cpu"


def cleanup_device(device: str) -> None:
    """
    Best-effort device-specific cleanup to release memory.
    """
    try:
        import torch
    except Exception:
        return

    device = (device or "cpu").lower()

    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS exposes empty_cache/synchronize for memory pressure relief
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except Exception:
        # Keep silent; cleanup is best-effort.
        pass
