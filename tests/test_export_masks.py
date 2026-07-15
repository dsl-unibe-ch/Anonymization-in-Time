"""Tests for SAM3 mask-source resolution in ait.export_video."""

import tempfile
import unittest
from pathlib import Path

from ait.export_video import resolve_sam3_file


class ResolveSam3FileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, name):
        (self.dir / name).write_bytes(b"")

    def test_explicit_original(self):
        self._touch("sam3.pkl")
        self._touch("sam3_circular.pkl")
        self.assertEqual(
            resolve_sam3_file(self.dir, "original"), self.dir / "sam3.pkl"
        )

    def test_explicit_circular(self):
        self._touch("sam3.pkl")
        self._touch("sam3_circular.pkl")
        self.assertEqual(
            resolve_sam3_file(self.dir, "circular"),
            self.dir / "sam3_circular.pkl",
        )

    def test_auto_defaults_to_original(self):
        self._touch("sam3.pkl")
        self._touch("sam3_circular.pkl")
        self.assertEqual(
            resolve_sam3_file(self.dir, "auto"), self.dir / "sam3.pkl"
        )

    def test_auto_follows_recorded_viewer_choice(self):
        self._touch("sam3.pkl")
        self._touch("sam3_circular.pkl")
        (self.dir / "mask_choice.txt").write_text("circular\n")
        self.assertEqual(
            resolve_sam3_file(self.dir, "auto"),
            self.dir / "sam3_circular.pkl",
        )

    def test_auto_ignores_recorded_choice_if_file_missing(self):
        self._touch("sam3.pkl")
        (self.dir / "mask_choice.txt").write_text("circular\n")
        self.assertEqual(
            resolve_sam3_file(self.dir, "auto"), self.dir / "sam3.pkl"
        )

    def test_auto_uses_circular_when_only_variant(self):
        self._touch("sam3_circular.pkl")
        self.assertEqual(
            resolve_sam3_file(self.dir, "auto"),
            self.dir / "sam3_circular.pkl",
        )


if __name__ == "__main__":
    unittest.main()
