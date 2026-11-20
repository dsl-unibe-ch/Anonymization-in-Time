"""Steady alterego placement and colour across frames."""

import unittest

import numpy as np

from ait.obfuscation import track_text_color
from ait.ocr.steady import stabilized_text_boxes, steady_ocr_boxes, hidden_box


def _ann(track, box, show=True):
    return {"track_id": track, "bbox": tuple(int(v) for v in box), "to_show": show}


class StabilizedTextBoxesTest(unittest.TestCase):
    def _scene(self, scroll, header=True, seed=0):
        """Eight scrolling words (a name split in two, plus chat-list rows)
        with +-1 px OCR noise on every edge, and optionally a static chat
        title, over len(scroll) frames."""
        rng = np.random.default_rng(seed)
        words = {1: (100, 300, 160, 320), 2: (165, 300, 220, 320)}
        words.update({3 + i: (100, 400 + 70 * i, 190, 420 + 70 * i) for i in range(6)})
        ocr = {}
        for f, s in enumerate(scroll):
            anns = []
            for track, (x1, y1, x2, y2) in words.items():
                n = rng.integers(-1, 2, 4)
                anns.append(_ann(track, (x1 + n[0], y1 - s + n[1], x2 + n[2], y2 - s + n[3])))
            if header:
                n = rng.integers(-1, 2, 4)
                anns.append(_ann(9, (60 + n[0], 20 + n[1], 140 + n[2], 40 + n[3])))
            ocr[f] = anns
        return ocr

    def _steps(self, ocr, boxes, track, page, raw=False):
        """Frame-to-frame edge changes of (box - true scroll) for one track."""
        rel = []
        for f in sorted(ocr):
            ann = next(a for a in ocr[f] if a["track_id"] == track)
            b = ann["bbox"] if raw else boxes[id(ann)]
            rel.append([b[0], b[1] + page[f], b[2], b[3] + page[f]])
        return np.abs(np.diff(np.array(rel), axis=0))

    def _check_steady(self, ocr, page, tracks, rare=0.05, gain=2.0):
        boxes = stabilized_text_boxes(ocr)
        for track in tracks:
            steps = self._steps(ocr, boxes, track, page)
            raw = self._steps(ocr, boxes, track, page, raw=True)
            self.assertLessEqual(steps.max(), 2)
            self.assertLessEqual(np.mean(steps >= 2), rare)
            self.assertLess(np.mean(steps >= 1), np.mean(raw >= 1) / gain)

    def test_static_text_stops_wobbling(self):
        self._check_steady(self._scene([0] * 30), [0] * 30, (1, 2, 3, 9))

    def test_follows_uneven_scrolling_without_lag(self):
        # flick: accelerate, fast, decelerate, stop
        speeds = [0] * 5 + [2, 6, 14, 25, 30, 30, 25, 16, 9, 4, 2, 1] + [0] * 8
        scroll = [int(s) for s in np.cumsum(speeds)]
        # weaker during a flick (the scroll estimate is noisy too), still a big gain
        self._check_steady(self._scene(scroll), scroll, (1, 2, 3), rare=0.10, gain=1.5)

    def test_static_title_stays_put_while_page_scrolls(self):
        scroll = list(np.cumsum([0] * 5 + [10] * 15 + [0] * 5))
        ocr = self._scene(scroll)
        boxes = stabilized_text_boxes(ocr)
        tops = [boxes[id(next(a for a in ocr[f] if a["track_id"] == 9))][1] for f in ocr]
        self.assertLessEqual(max(tops) - min(tops), 1)

    def test_hidden_and_untracked_boxes_skipped(self):
        ocr = {0: [_ann(None, (0, 0, 10, 10)), _ann(1, (0, 0, 10, 10), show=False)]}
        self.assertEqual(stabilized_text_boxes(ocr), {})


class SteadyOcrBoxesTest(unittest.TestCase):
    _scene = StabilizedTextBoxesTest._scene

    def test_pipeline_step_keeps_detection_and_hides_both(self):
        ocr = self._scene([0] * 20)
        ocr[3].append(_ann(None, (5, 5, 30, 15)))          # untracked: untouched
        out = steady_ocr_boxes(ocr)
        moved = 0
        for f in ocr:
            for before, after in zip(ocr[f], out[f]):
                self.assertIsNot(before, after)            # input not modified
                if before["track_id"] is None:
                    self.assertEqual(after, before)
                    continue
                det = after.get("detected_bbox", after["bbox"])
                self.assertEqual(tuple(det), tuple(before["bbox"]))
                moved += "detected_bbox" in after
                hx1, hy1, hx2, hy2 = hidden_box(after)
                for b in (after["bbox"], det):             # union covers both
                    self.assertTrue(hx1 <= b[0] and hy1 <= b[1] and b[2] <= hx2 and b[3] <= hy2)
        self.assertGreater(moved, 0)

    def test_hidden_boxes_are_steadied_too(self):
        ocr = self._scene([0] * 20)
        for f in ocr:
            for a in ocr[f]:
                a["to_show"] = False
        self.assertTrue(any("detected_bbox" in a for f in ocr for a in steady_ocr_boxes(ocr)[f]))


class TrackTextColorTest(unittest.TestCase):
    def test_one_colour_across_noisy_frames(self):
        # grey right at a snapping boundary (144 = between 128 and 160)
        rng = np.random.default_rng(0)
        frames = [np.array([144, 144, 144]) + rng.normal(0, 3, 3) for _ in range(50)]
        self.assertIn(track_text_color(frames), {(128, 128, 128), (160, 160, 160)})
        # a coloured text keeps its hue
        b, g, r = track_text_color([np.array([60, 200, 40])] * 5)
        self.assertTrue(g > b and g > r)
        self.assertEqual(track_text_color([None, None], (1, 2, 3)), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
