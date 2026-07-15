"""Tests for ait.export_queue (pure controller, fake subprocesses)."""

import tempfile
import unittest
from pathlib import Path

from ait.export_queue import (
    ExportQueueController,
    ExportQueueError,
    ItemStatus,
    build_export_command,
    validate_processed_folder,
)


class FakeProcess:
    """Minimal stand-in for subprocess.Popen used by the controller."""

    def __init__(self, exit_code=0, polls_before_exit=0):
        self._exit_code = exit_code
        self._remaining = polls_before_exit
        self.returncode = None
        self.terminated = False

    def poll(self):
        if self.terminated:
            return self.returncode
        if self._remaining > 0:
            self._remaining -= 1
            return None
        self.returncode = self._exit_code
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self.returncode = -15


class FakeRunner:
    """Records launched commands and hands back queued fake processes."""

    def __init__(self, outcomes):
        # outcomes: list of FakeProcess or Exception instances to raise
        self._outcomes = list(outcomes)
        self.commands = []

    def __call__(self, cmd):
        self.commands.append(cmd)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_processed_folder(base, name, marker="state.pkl"):
    folder = Path(base) / name
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    (folder / marker).write_bytes(b"")
    return folder


class ValidateFolderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_valid_with_state(self):
        folder = make_processed_folder(self.base, "vid", marker="state.pkl")
        ok, reason = validate_processed_folder(folder)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_valid_with_ocr_or_sam3(self):
        for marker in ("ocr.pkl", "sam3.pkl"):
            folder = make_processed_folder(self.base, f"v_{marker}", marker=marker)
            ok, _ = validate_processed_folder(folder)
            self.assertTrue(ok, marker)

    def test_missing_frames(self):
        folder = self.base / "noframes"
        folder.mkdir()
        (folder / "state.pkl").write_bytes(b"")
        ok, reason = validate_processed_folder(folder)
        self.assertFalse(ok)
        self.assertIn("frames", reason)

    def test_missing_annotations(self):
        folder = self.base / "onlyframes"
        (folder / "frames").mkdir(parents=True)
        ok, reason = validate_processed_folder(folder)
        self.assertFalse(ok)
        self.assertIn("state.pkl", reason)


class AddValidationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.controller = ExportQueueController(runner=FakeRunner([]))

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_invalid_folder_raises(self):
        bad = self.base / "empty"
        bad.mkdir()
        with self.assertRaises(ExportQueueError):
            self.controller.add(bad, self.base / "out.mp4", 51)

    def test_add_stores_blur_and_paths(self):
        folder = make_processed_folder(self.base, "vid")
        item = self.controller.add(folder, self.base / "out.mp4", 50)
        # Even blur strengths are coerced to int as supplied (odd handling is UI).
        self.assertEqual(item.blur_strength, 50)
        self.assertEqual(item.status, ItemStatus.PENDING)
        self.assertEqual(Path(item.source_dir), folder)

    def test_duplicate_source_rejected(self):
        folder = make_processed_folder(self.base, "vid")
        self.controller.add(folder, self.base / "a.mp4", 51)
        with self.assertRaises(ExportQueueError):
            self.controller.add(folder, self.base / "b.mp4", 51)

    def test_duplicate_output_rejected(self):
        f1 = make_processed_folder(self.base, "vid1")
        f2 = make_processed_folder(self.base, "vid2")
        self.controller.add(f1, self.base / "shared.mp4", 51)
        with self.assertRaises(ExportQueueError):
            self.controller.add(f2, self.base / "shared.mp4", 51)

    def test_case_only_output_difference_is_rejected_portably(self):
        f1 = make_processed_folder(self.base, "vid1")
        f2 = make_processed_folder(self.base, "vid2")
        self.controller.add(f1, self.base / "shared.mp4", 51)
        with self.assertRaises(ExportQueueError):
            self.controller.add(f2, self.base / "SHARED.MP4", 51)


class SequencingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _add(self, controller, n):
        items = []
        for i in range(n):
            folder = make_processed_folder(self.base, f"vid{i}")
            items.append(controller.add(folder, self.base / f"out{i}.mp4", 51))
        return items

    def test_one_subprocess_at_a_time(self):
        procs = [FakeProcess(0, polls_before_exit=2) for _ in range(3)]
        runner = FakeRunner(procs)
        controller = ExportQueueController(runner=runner)
        self._add(controller, 3)

        controller.start()
        # Only the first job launched.
        self.assertEqual(len(runner.commands), 1)
        self.assertTrue(controller.is_active())

        # Poll while first still running: no new launch.
        controller.poll()
        self.assertEqual(len(runner.commands), 1)
        controller.poll()
        self.assertEqual(len(runner.commands), 1)
        # Now the first exits -> second launches.
        controller.poll()
        self.assertEqual(len(runner.commands), 2)

    def test_success_sequence_all_succeed(self):
        runner = FakeRunner([FakeProcess(0), FakeProcess(0), FakeProcess(0)])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 3)

        controller.start()
        for _ in range(6):  # more polls than needed
            controller.poll()

        self.assertFalse(controller.is_active())
        self.assertEqual(len(runner.commands), 3)
        for item in items:
            self.assertEqual(item.status, ItemStatus.SUCCEEDED)
            self.assertEqual(item.exit_code, 0)
        self.assertEqual(controller.summary()[ItemStatus.SUCCEEDED], 3)

    def test_nonzero_exit_fails_only_that_item(self):
        runner = FakeRunner([FakeProcess(0), FakeProcess(2), FakeProcess(0)])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 3)

        controller.start()
        for _ in range(6):
            controller.poll()

        self.assertEqual(items[0].status, ItemStatus.SUCCEEDED)
        self.assertEqual(items[1].status, ItemStatus.FAILED)
        self.assertEqual(items[1].exit_code, 2)
        self.assertEqual(items[2].status, ItemStatus.SUCCEEDED)
        self.assertEqual(len(runner.commands), 3)

    def test_launch_failure_fails_only_that_item_and_continues(self):
        runner = FakeRunner([
            RuntimeError("boom"),   # first fails to even launch
            FakeProcess(0),         # second launches fine
        ])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 2)

        controller.start()  # first launch raises, skips to second
        # Second item is now the active one.
        self.assertTrue(controller.is_active())
        self.assertEqual(items[0].status, ItemStatus.FAILED)
        self.assertIn("Failed to launch", items[0].error)
        self.assertEqual(items[1].status, ItemStatus.RUNNING)

        controller.poll()  # second finishes
        self.assertEqual(items[1].status, ItemStatus.SUCCEEDED)
        self.assertFalse(controller.is_active())

    def test_start_while_active_is_noop(self):
        runner = FakeRunner([FakeProcess(0, polls_before_exit=1), FakeProcess(0)])
        controller = ExportQueueController(runner=runner)
        self._add(controller, 2)
        controller.start()
        self.assertEqual(len(runner.commands), 1)
        # A second start must not launch a concurrent subprocess.
        self.assertFalse(controller.start())
        self.assertEqual(len(runner.commands), 1)

    def test_cancel_active_continues_pending(self):
        first = FakeProcess(0, polls_before_exit=5)  # long-running, gets cancelled
        runner = FakeRunner([
            first,
            FakeProcess(0),                          # next one runs after cancel
        ])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 2)

        controller.start()
        self.assertTrue(controller.is_active())
        controller.cancel_active(continue_pending=True)

        self.assertEqual(items[0].status, ItemStatus.CANCELLING)
        self.assertTrue(first.terminated)
        # terminate() is asynchronous: the second job must not launch until a
        # poll has observed that the first process actually exited.
        self.assertEqual(len(runner.commands), 1)
        self.assertTrue(controller.is_active())
        controller.poll()
        self.assertEqual(items[0].status, ItemStatus.CANCELLED)
        self.assertEqual(len(runner.commands), 2)
        self.assertEqual(items[1].status, ItemStatus.RUNNING)
        controller.poll()
        self.assertEqual(items[1].status, ItemStatus.SUCCEEDED)

    def test_cancel_active_without_continue_stops(self):
        runner = FakeRunner([FakeProcess(0, polls_before_exit=5), FakeProcess(0)])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 2)
        controller.start()
        controller.cancel_active(continue_pending=False)
        self.assertEqual(items[0].status, ItemStatus.CANCELLING)
        self.assertTrue(controller.is_active())
        controller.poll()
        self.assertEqual(items[0].status, ItemStatus.CANCELLED)
        self.assertFalse(controller.is_active())
        self.assertEqual(items[1].status, ItemStatus.PENDING)
        self.assertEqual(len(runner.commands), 1)

    def test_shutdown_terminates_active_and_cancels_pending(self):
        active = FakeProcess(0, polls_before_exit=5)
        runner = FakeRunner([active, FakeProcess(0)])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 2)
        controller.start()

        controller.shutdown()

        self.assertTrue(active.terminated)
        self.assertEqual(items[0].status, ItemStatus.CANCELLED)
        self.assertEqual(items[1].status, ItemStatus.CANCELLED)
        self.assertFalse(controller.is_active())
        # No further launches happen after shutdown.
        controller.poll()
        self.assertEqual(len(runner.commands), 1)

    def test_remove_and_clear_pending(self):
        runner = FakeRunner([FakeProcess(0, polls_before_exit=5)])
        controller = ExportQueueController(runner=runner)
        items = self._add(controller, 3)

        controller.start()  # items[0] running
        # Cannot remove the running item.
        self.assertFalse(controller.remove(items[0]))
        # Can remove a pending item.
        self.assertTrue(controller.remove(items[1]))
        # Clear the rest of the pending queue.
        cleared = controller.clear_pending()
        self.assertEqual(cleared, 1)
        self.assertFalse(controller.has_pending())
        self.assertTrue(controller.is_active())


class CommandBuildTest(unittest.TestCase):
    def test_command_uses_export_video_module(self):
        controller = ExportQueueController(runner=FakeRunner([]), python_executable="py")
        with tempfile.TemporaryDirectory() as base:
            folder = make_processed_folder(base, "vid")
            item = controller.add(folder, Path(base) / "out.mp4", 33)
            cmd = build_export_command(item, "py")
        self.assertEqual(cmd[:4], ["py", "-m", "ait.export_video", "--video_dir"])
        self.assertIn("--output", cmd)
        self.assertIn("--blur_strength", cmd)
        self.assertIn("33", cmd)


if __name__ == "__main__":
    unittest.main()
