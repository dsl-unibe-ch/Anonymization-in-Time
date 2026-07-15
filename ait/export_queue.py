"""Sequential export-queue controller for the AiT launcher.

The controller orchestrates one ``python -m ait.export_video`` subprocess at a
time. It is intentionally free of any Tkinter or OpenCV/torch imports so it can
be driven by a Tk ``after`` polling loop in the launcher *and* unit-tested with
fake processes. The UI owns the polling cadence; this module never blocks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ItemStatus:
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})


class ExportQueueError(Exception):
    """Raised when an item cannot be added (validation / duplicate)."""


@dataclass
class ExportItem:
    """A single queued export job.

    Records the validated processed-video folder, the explicit output path,
    the blur strength and the SAM3 mask source captured at enqueue time, plus
    mutable status fields.
    """

    source_dir: Path
    output_path: Path
    blur_strength: int
    sam3_source: str = "auto"
    delete_source_after: bool = False
    status: str = ItemStatus.PENDING
    exit_code: Optional[int] = None
    error: Optional[str] = None
    _id: int = field(default=0, repr=False)


def validate_processed_folder(folder):
    """Validate that ``folder`` looks like a processed-video folder.

    Requires a ``frames/`` subdirectory and at least one of ``state.pkl``,
    ``ocr.pkl``, ``sam3.pkl`` or ``sam3_circular.pkl``. Returns ``(ok,
    reason)`` where ``reason`` is ``None`` on success and a human-readable
    message on failure.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return False, "The selected path is not a folder."
    if not (folder / "frames").is_dir():
        return False, "The folder does not contain a 'frames' subdirectory."
    for name in ("state.pkl", "ocr.pkl", "sam3.pkl", "sam3_circular.pkl"):
        if (folder / name).exists():
            return True, None
    return False, (
        "The folder does not contain state.pkl, ocr.pkl, sam3.pkl, "
        "or sam3_circular.pkl."
    )


def discover_pipeline_folders(root) -> list:
    """Find processed-video folders at or under ``root``.

    Returns a deterministically ordered list of directories that pass
    ``validate_processed_folder``. If ``root`` itself is a pipeline folder it
    is the only result. Otherwise the tree is walked and each matched folder is
    returned without descending further into it (so a mirrored output tree like
    ``base/a/b/clip`` yields the ``clip`` folders, not their ``frames/``).
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames.sort()  # deterministic traversal order
        current = Path(dirpath)
        ok, _ = validate_processed_folder(current)
        if ok:
            found.append(current)
            dirnames[:] = []  # matched: do not descend into a pipeline folder
    return found


def _path_key(path) -> str:
    """Conservative, cross-platform identity for a queue path.

    Queue files may be prepared on one platform and opened on another, and the
    default macOS and Windows filesystems are case-insensitive. Case-folding on
    every platform deliberately rejects ``out.mp4``/``OUT.mp4`` as conflicting
    even on a case-sensitive Linux volume rather than allowing a non-portable
    queue that can overwrite after moving it.
    """
    return os.path.normpath(os.path.abspath(str(path))).casefold()


def build_export_command(item: ExportItem, python_executable: str = None) -> list:
    """Build the ``python -m ait.export_video ...`` command for an item."""
    python_executable = python_executable or sys.executable
    return [
        python_executable,
        "-m",
        "ait.export_video",
        "--video_dir",
        str(item.source_dir),
        "--output",
        str(item.output_path),
        "--blur_strength",
        str(item.blur_strength),
        "--masks",
        item.sam3_source,
    ]


def default_runner(cmd: list):
    """Launch the export subprocess (real runner used by the launcher)."""
    creationflags = (
        subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    return subprocess.Popen(cmd, creationflags=creationflags)


class ExportQueueController:
    """Drive queued export jobs sequentially without blocking the UI thread.

    ``poll()`` is expected to be called periodically (e.g. from Tk ``after``).
    At most one subprocess runs at a time; the controller advances to the next
    pending item after the active one exits, whether it succeeded, exited
    non-zero, or failed to launch.
    """

    def __init__(self, runner=None, python_executable=None, command_builder=None):
        self.items = []
        self._runner = runner or default_runner
        self._python = python_executable or sys.executable
        self._build_cmd = command_builder or build_export_command
        self.active_item = None
        self.active_proc = None
        self._shutting_down = False
        self._cancel_requested = False
        self._continue_after_cancel = False
        self._next_id = 1

    # -- queue mutation -------------------------------------------------

    def add(self, source_dir, output_path, blur_strength,
            sam3_source="auto", delete_source_after=False) -> ExportItem:
        """Validate and enqueue a job, rejecting duplicates.

        Raises ``ExportQueueError`` with a user-facing message when the folder
        is invalid, the source folder is already queued, or the output path is
        already represented in the queue (pending/running/completed).

        When ``delete_source_after`` is True the processed-video folder is
        removed after the export exits successfully (see ``poll``).
        """
        source_dir = Path(source_dir)
        output_path = Path(output_path)
        if sam3_source not in ("auto", "original", "circular"):
            raise ExportQueueError(
                f"Invalid SAM3 mask source: {sam3_source!r}."
            )

        ok, reason = validate_processed_folder(source_dir)
        if not ok:
            raise ExportQueueError(reason)

        src_key = _path_key(source_dir)
        out_key = _path_key(output_path)
        for existing in self.items:
            if _path_key(existing.source_dir) == src_key:
                raise ExportQueueError(
                    "This processed folder is already in the queue."
                )
            if _path_key(existing.output_path) == out_key:
                raise ExportQueueError(
                    "Another queued job already targets this output path."
                )

        item = ExportItem(
            source_dir=source_dir,
            output_path=output_path,
            blur_strength=int(blur_strength),
            sam3_source=sam3_source,
            delete_source_after=bool(delete_source_after),
            _id=self._next_id,
        )
        self._next_id += 1
        self.items.append(item)
        return item

    def remove(self, item: ExportItem) -> bool:
        """Remove a *pending* item. Running/finished items are never removed."""
        if item.status != ItemStatus.PENDING:
            return False
        try:
            self.items.remove(item)
        except ValueError:
            return False
        return True

    def clear_pending(self) -> int:
        """Drop all pending items, leaving running/finished ones intact."""
        before = len(self.items)
        self.items = [
            it for it in self.items if it.status != ItemStatus.PENDING
        ]
        return before - len(self.items)

    # -- queue state ----------------------------------------------------

    def is_active(self) -> bool:
        return self.active_proc is not None

    def has_pending(self) -> bool:
        return any(it.status == ItemStatus.PENDING for it in self.items)

    def summary(self) -> Counter:
        """Aggregate counts by status."""
        return Counter(it.status for it in self.items)

    def summary_text(self) -> str:
        counts = self.summary()
        total = len(self.items)
        parts = [f"{total} job(s)"]
        for status in (
            ItemStatus.RUNNING,
            ItemStatus.CANCELLING,
            ItemStatus.PENDING,
            ItemStatus.SUCCEEDED,
            ItemStatus.FAILED,
            ItemStatus.CANCELLED,
        ):
            if counts.get(status):
                parts.append(f"{counts[status]} {status}")
        return ", ".join(parts)

    # -- execution ------------------------------------------------------

    def start(self) -> bool:
        """Start the queue if idle. No-op while a job is already running."""
        if self.active_proc is not None:
            return False
        self._shutting_down = False
        return self._launch_next()

    def _launch_next(self) -> bool:
        """Launch the next pending item; skip items that fail to launch."""
        for item in self.items:
            if item.status != ItemStatus.PENDING:
                continue
            cmd = self._build_cmd(item, self._python)
            try:
                proc = self._runner(cmd)
            except Exception as exc:  # launch failure fails only this item
                item.status = ItemStatus.FAILED
                item.error = f"Failed to launch export: {exc}"
                continue
            item.status = ItemStatus.RUNNING
            self.active_item = item
            self.active_proc = proc
            self._cancel_requested = False
            self._continue_after_cancel = False
            return True
        self.active_item = None
        self.active_proc = None
        return False

    def poll(self) -> None:
        """Non-blocking check on the active process; advance when it exits."""
        if self.active_proc is None:
            return
        returncode = self.active_proc.poll()
        if returncode is None:
            return  # still running

        item = self.active_item
        if item is None:  # defensive: active process ownership must include an item
            self.active_proc = None
            self._cancel_requested = False
            self._continue_after_cancel = False
            return
        item.exit_code = returncode
        was_cancelled = self._cancel_requested
        continue_after_cancel = self._continue_after_cancel
        if was_cancelled:
            item.status = ItemStatus.CANCELLED
            item.error = "Cancelled by user."
        elif returncode == 0:
            item.status = ItemStatus.SUCCEEDED
            if item.delete_source_after:
                self._delete_source(item)
        else:
            item.status = ItemStatus.FAILED
            item.error = f"Export exited with code {returncode}."
        self.active_item = None
        self.active_proc = None
        self._cancel_requested = False
        self._continue_after_cancel = False

        if not self._shutting_down and (not was_cancelled or continue_after_cancel):
            self._launch_next()

    def cancel_active(self, continue_pending: bool = True) -> bool:
        """Terminate the running job, marking it cancelled.

        By default remaining pending jobs continue (ordinary cancel). Pass
        ``continue_pending=False`` to stop after cancelling the active job.
        """
        if self.active_proc is None:
            return False
        if self._cancel_requested:
            return False
        item = self.active_item
        if item is None:  # defensive consistency guard
            return False
        self._terminate(self.active_proc)
        # ``terminate()`` is asynchronous. Keep ownership of this process until
        # poll() observes its exit; launching the next item here could briefly
        # run two exporters concurrently.
        item.status = ItemStatus.CANCELLING
        item.error = "Cancellation requested."
        self._cancel_requested = True
        self._continue_after_cancel = continue_pending
        return True

    def shutdown(self) -> None:
        """Terminate any active job and cancel all pending work (no advance).

        Used on launcher close so no export subprocess or hidden pending queue
        is left orphaned.
        """
        self._shutting_down = True
        if self.active_proc is not None:
            self._terminate_and_wait(self.active_proc)
            if self.active_item is not None:
                self.active_item.status = ItemStatus.CANCELLED
                self.active_item.error = "Cancelled because the launcher closed."
            self.active_item = None
            self.active_proc = None
            self._cancel_requested = False
            self._continue_after_cancel = False
        for item in self.items:
            if item.status == ItemStatus.PENDING:
                item.status = ItemStatus.CANCELLED

    @staticmethod
    def _delete_source(item: ExportItem) -> None:
        """Remove the processed-video folder after a successful export.

        Deletion failures are surfaced on the item but never turn a successful
        export into a failure — the video was still produced.
        """
        try:
            shutil.rmtree(item.source_dir)
        except Exception as exc:  # keep the SUCCEEDED status; just note it
            item.error = (
                f"Export succeeded but could not delete the source folder: {exc}"
            )

    @staticmethod
    def _terminate(proc) -> None:
        try:
            proc.terminate()
        except Exception:
            pass

    @staticmethod
    def _terminate_and_wait(proc) -> None:
        """Best-effort bounded shutdown so launcher close leaves no exporter."""
        ExportQueueController._terminate(proc)
        wait = getattr(proc, "wait", None)
        if wait is None:
            return
        try:
            wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                wait(timeout=2)
            except Exception:
                pass
        except Exception:
            pass
