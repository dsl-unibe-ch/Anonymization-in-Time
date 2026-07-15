"""Tests for ait.video_discovery (pure, no ML/video imports)."""

import tempfile
import unittest
from pathlib import Path

from ait.video_discovery import (
    SUPPORTED_VIDEO_EXTENSIONS,
    discover_videos,
    is_supported_video,
    plan_output_paths,
)


class DiscoverVideosTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, relpath):
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def _rel(self, paths):
        return [str(Path(p).relative_to(self.root)) for p in paths]

    def test_nested_and_subnested_folders(self):
        self._touch("top.mp4")
        self._touch("a/one.mp4")
        self._touch("a/b/two.mov")
        self._touch("a/b/c/three.mkv")
        found = discover_videos(self.root)
        self.assertEqual(
            self._rel(found),
            [
                str(Path("a/b/c/three.mkv")),
                str(Path("a/b/two.mov")),
                str(Path("a/one.mp4")),
                "top.mp4",
            ],
        )

    def test_mixed_case_extensions(self):
        self._touch("A.MP4")
        self._touch("b.Mov")
        self._touch("c.WEBM")
        found = discover_videos(self.root)
        self.assertEqual({p.name for p in found}, {"A.MP4", "b.Mov", "c.WEBM"})

    def test_ignores_unsupported_files(self):
        self._touch("keep.mp4")
        self._touch("notes.txt")
        self._touch("image.png")
        self._touch("archive.zip")
        self._touch("nested/readme.md")
        found = discover_videos(self.root)
        self.assertEqual([p.name for p in found], ["keep.mp4"])

    def test_each_path_returned_once(self):
        self._touch("dir/clip.mp4")
        self._touch("dir/clip2.mp4")
        found = discover_videos(self.root)
        self.assertEqual(len(found), len(set(found)))
        self.assertEqual(len(found), 2)

    def test_relative_path_deterministic_order(self):
        # Insertion order deliberately scrambled relative to expected order.
        self._touch("z/last.mp4")
        self._touch("a/second.mp4")
        self._touch("a/first.mp4")
        self._touch("m.mp4")
        found = discover_videos(self.root)
        self.assertEqual(
            self._rel(found),
            [
                str(Path("a/first.mp4")),
                str(Path("a/second.mp4")),
                "m.mp4",
                str(Path("z/last.mp4")),
            ],
        )
        # Order is stable across repeated calls.
        self.assertEqual(self._rel(found), self._rel(discover_videos(self.root)))

    def test_no_videos_returns_empty(self):
        self._touch("readme.txt")
        self.assertEqual(discover_videos(self.root), [])

    def test_is_supported_video(self):
        self.assertTrue(is_supported_video("x.MP4"))
        self.assertTrue(is_supported_video(Path("y.webm")))
        self.assertFalse(is_supported_video("z.txt"))
        # Every documented extension is covered.
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"):
            self.assertIn(ext, SUPPORTED_VIDEO_EXTENSIONS)


class PlanOutputPathsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, relpath):
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def test_mirrors_source_tree(self):
        paths = [
            self._touch("group-a/alpha.mp4"),
            self._touch("group-b/nested/beta.mp4"),
            self._touch("gamma.mp4"),
        ]
        assignments = dict(plan_output_paths(paths, self.root))
        self.assertEqual(assignments[paths[0]], "group-a/alpha")
        self.assertEqual(assignments[paths[1]], "group-b/nested/beta")
        self.assertEqual(assignments[paths[2]], "gamma")

    def test_same_stem_different_folders_keep_their_paths(self):
        p1 = self._touch("group-a/clip.mp4")
        p2 = self._touch("group-b/clip.mp4")
        assignments = dict(plan_output_paths([p1, p2], self.root))
        self.assertEqual(assignments[p1], "group-a/clip")
        self.assertEqual(assignments[p2], "group-b/clip")

    def test_same_stem_at_different_depths_never_collide(self):
        p1 = self._touch("clip.mp4")
        p2 = self._touch("a/clip.mp4")
        p3 = self._touch("a/b/clip.mp4")
        names = [name for _, name in plan_output_paths([p1, p2, p3], self.root)]
        self.assertEqual(len(set(names)), 3)

    def test_same_stem_different_extension_disambiguated(self):
        # Same folder, same stem, different extensions -> subpaths would
        # collide; result must still be distinct.
        p1 = self._touch("clip.mp4")
        p2 = self._touch("clip.mov")
        names = [name for _, name in plan_output_paths([p1, p2], self.root)]
        self.assertEqual(len(set(n.casefold() for n in names)), 2)

    def test_path_components_are_filesystem_safe(self):
        p1 = self._touch("odd dir/weird:name.mp4")
        names = [name for _, name in plan_output_paths([p1], self.root)]
        for name in names:
            # Slash is the allowed subpath separator; each component is safe.
            for component in name.split("/"):
                for bad in '\\:*?"<>| ':
                    self.assertNotIn(bad, component)

    def test_deterministic_regardless_of_input_order(self):
        p1 = self._touch("a/clip.mp4")
        p2 = self._touch("b/clip.mp4")
        first = dict(plan_output_paths([p1, p2], self.root))
        second = dict(plan_output_paths([p2, p1], self.root))
        self.assertEqual(first[p1], second[p1])
        self.assertEqual(first[p2], second[p2])


if __name__ == "__main__":
    unittest.main()
