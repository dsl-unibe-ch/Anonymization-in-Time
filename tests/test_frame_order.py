"""Regression tests for frame ordering and export frame rate.

Frames are extracted as ``f"{idx:04d}.jpg"``, so from index 10000 on the names
get five digits and alphabetical order breaks (``10000.jpg`` sorts before
``1001.jpg``). SAM3 used to key its masks by the position in that alphabetical
list, which put profile-picture masks seconds to minutes away from their frames
in videos longer than 9,999 frames (and on the wrong frames whenever
``frame_step > 1``). Export also wrote frames in alphabetical order and fell back
to 30 fps unless the source video sat next to the output folder.
"""

import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ait.frame_files import (frame_number, list_frame_files, order_results_by_frame,
                             read_video_info, write_video_info)

REPO = Path(__file__).resolve().parents[1]


def _load_module(relpath, name):
    """Load one module file without importing its package (keeps these tests ML-free)."""
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch_frames(folder, indices, ext=".jpg"):
    for i in indices:
        (folder / f"{i:04d}{ext}").write_bytes(b"")


class FrameNumberTest(unittest.TestCase):
    def test_reads_index_from_name(self):
        self.assertEqual(frame_number("0042.jpg"), 42)
        self.assertEqual(frame_number(Path("x") / "10000.jpg"), 10000)

    def test_no_digits(self):
        self.assertIsNone(frame_number("cover.jpg"))


class ListFrameFilesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_numeric_order_past_9999(self):
        indices = [0, 999, 1000, 1001, 9999, 10000, 10001, 12000]
        _touch_frames(self.dir, indices)
        got = [frame_number(p) for p in list_frame_files(self.dir)]
        self.assertEqual(got, indices)

    def test_extension_filter_and_case(self):
        _touch_frames(self.dir, [1, 2])
        (self.dir / "0003.JPG").write_bytes(b"")
        (self.dir / "video_info.json").write_text("{}")
        got = [frame_number(p) for p in list_frame_files(self.dir, (".jpg",))]
        self.assertEqual(got, [1, 2, 3])


class OrderResultsTest(unittest.TestCase):
    def test_reorders_and_rekeys_legacy_results(self):
        # Legacy SAM3 cache: alphabetical file order, keyed by list position.
        names = sorted(f"{i:04d}.jpg" for i in (999, 1000, 1001, 10000, 10001))
        legacy = [(pos, {"pos": pos}, Path(n)) for pos, n in enumerate(names)]
        fixed, changed = order_results_by_frame(legacy)
        self.assertTrue(changed)
        self.assertEqual([k for k, _, _ in fixed], [999, 1000, 1001, 10000, 10001])
        # each result stays attached to its own image
        for k, res, path in fixed:
            self.assertEqual(frame_number(path), k)
            self.assertEqual(names[res["pos"]], path.name)

    def test_already_ordered_results_unchanged(self):
        ok = [(i, {}, Path(f"{i:04d}.jpg")) for i in (0, 5, 10)]
        fixed, changed = order_results_by_frame(ok)
        self.assertFalse(changed)
        self.assertEqual(fixed, ok)

    def test_legacy_frame_step_results_are_rekeyed(self):
        # frame_step=5: files 0000, 0005, 0010 were stored under keys 0, 1, 2.
        legacy = [(pos, {}, Path(f"{i:04d}.jpg")) for pos, i in enumerate((0, 5, 10))]
        fixed, changed = order_results_by_frame(legacy)
        self.assertTrue(changed)
        self.assertEqual([k for k, _, _ in fixed], [0, 5, 10])


class Sam3UnifiedKeysTest(unittest.TestCase):
    def test_keys_are_frame_numbers_and_tracks_use_list_positions(self):
        fmt = _load_module("ait/segmentation/format.py", "ait_segmentation_format_under_test")
        mask = np.ones((4, 4), dtype=bool)
        res = lambda: {"boxes": [np.array([0, 0, 3, 3], np.float32)], "masks": [mask], "scores": [0.9]}
        # frame_step=5: list positions 0, 1 hold frames 0 and 5
        all_results = [(0, res(), Path("missing/0000.jpg")), (5, res(), Path("missing/0005.jpg"))]
        tracks = [[(0, 0), (1, 0)]]  # tracks are built on list positions
        unified = fmt.convert_to_unified_dict(all_results, tracks)
        self.assertEqual(sorted(unified), [0, 5])
        self.assertEqual(unified[0][0]["track_id"], 0)
        self.assertEqual(unified[5][0]["track_id"], 0)


class VideoInfoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip(self):
        write_video_info(self.dir, source="clip.mp4", fps=59.94, frame_step=2, frame_count=100)
        info = read_video_info(self.dir)
        self.assertAlmostEqual(info["fps"], 59.94)
        self.assertEqual(info["frame_step"], 2)

    def test_missing_or_corrupt(self):
        self.assertIsNone(read_video_info(self.dir))
        (self.dir / "video_info.json").write_text("not json")
        self.assertIsNone(read_video_info(self.dir))


class ExportTest(unittest.TestCase):
    """End-to-end export on tiny synthetic frames (no ML)."""

    def setUp(self):
        from ait import export_video
        self.export_video = export_video
        self._tmp = tempfile.TemporaryDirectory()
        self.video_dir = Path(self._tmp.name) / "out" / "clip"
        self.frames = self.video_dir / "frames"
        self.frames.mkdir(parents=True)
        # brightness encodes the true frame order
        self.indices = [9998, 9999, 10000, 10001, 10002]
        for level, i in zip((20, 70, 120, 170, 220), self.indices):
            cv2.imwrite(str(self.frames / f"{i:04d}.jpg"), np.full((32, 32, 3), level, np.uint8))

    def tearDown(self):
        self._tmp.cleanup()

    def _read_back(self, path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        levels = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            levels.append(float(frame.mean()))
        cap.release()
        return fps, levels

    def test_frames_written_in_numeric_order(self):
        out = Path(self._tmp.name) / "result.mp4"
        self.export_video.export_anonymized_video(self.video_dir, out, fps=10)
        _, levels = self._read_back(out)
        self.assertEqual(len(levels), len(self.indices))
        self.assertEqual(levels, sorted(levels), "exported frames are out of order")

    def test_fps_from_recorded_video_info(self):
        write_video_info(self.frames, source="elsewhere/clip.mp4", fps=60.0, frame_step=1,
                         frame_count=len(self.indices))
        out = Path(self._tmp.name) / "result.mp4"
        self.export_video.export_anonymized_video(self.video_dir, out)
        fps, _ = self._read_back(out)
        self.assertAlmostEqual(fps, 60.0, places=1)

    def test_fps_accounts_for_frame_step(self):
        write_video_info(self.frames, source="clip.mp4", fps=60.0, frame_step=4, frame_count=20)
        self.assertAlmostEqual(self.export_video.resolve_export_fps(self.video_dir), 15.0)

    def test_explicit_fps_wins(self):
        write_video_info(self.frames, source="clip.mp4", fps=60.0, frame_step=1, frame_count=5)
        self.assertEqual(self.export_video.resolve_export_fps(self.video_dir, 25), 25)

    def test_sam3_masks_land_on_their_frame(self):
        # One mask, stored under real frame number 10001 (as the fixed pipeline does).
        h = w = 32
        mask = np.zeros((h, w), dtype=bool)
        mask[8:24, 8:24] = True
        sam3 = {10001: [{"bbox": (8, 8, 24, 24), "mask": mask, "source": "sam3",
                          "to_show": True, "score": 0.9, "track_id": 0}]}
        with open(self.video_dir / "sam3.pkl", "wb") as f:
            pickle.dump(sam3, f)
        # textured frames so the blur is measurable
        rng = np.random.default_rng(0)
        for i in self.indices:
            cv2.imwrite(str(self.frames / f"{i:04d}.png"), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        for p in self.frames.glob("*.jpg"):
            p.unlink()
        out = Path(self._tmp.name) / "masked.avi"
        self.export_video.export_anonymized_video(self.video_dir, out, fps=10, codec="FFV1",
                                                  blur_strength=15)
        cap = cv2.VideoCapture(str(out))
        blurred = []
        for i in self.indices:
            ok, frame = cap.read()
            src = cv2.imread(str(self.frames / f"{i:04d}.png"))
            blurred.append(float(np.abs(frame[8:24, 8:24].astype(int) - src[8:24, 8:24].astype(int)).mean()) > 5)
        cap.release()
        self.assertEqual(blurred, [False, False, False, True, False])


if __name__ == "__main__":
    unittest.main()
