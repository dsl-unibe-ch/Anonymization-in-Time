"""Tests for stable fake-name (alterego) font sizes across frames.

OCR boxes wobble by a pixel from frame to frame. Fitting the font to each
frame's box made a static name jump between sizes (e.g. 17 <-> 15 whenever
"Rossi"'s box was 45 instead of 46 px wide). The size is now chosen once per
OCR track.
"""

import unittest

from ait.utils import shared_alterego_font_sizes, track_alterego_font_sizes


def ann(track_id, width, alterego="Rossi", to_show=True, name="Smith John", parent=(0, 0, 100, 24)):
    return {"track_id": track_id, "bbox": (0, 0, width, 24), "alterego": alterego,
            "to_show": to_show, "name": name, "parent_box": parent}


def size_by_width(a):
    # stand-in for fit_alterego_font_size: 17 fits from 46 px, else 15
    return 17 if a["bbox"][2] >= 46 else 15


class TrackFontSizeTest(unittest.TestCase):
    def test_one_size_per_track_despite_width_wobble(self):
        widths = [46, 45, 46, 46, 45, 46, 46]            # box wobbles by a pixel
        frames = {i: [ann(7, w)] for i, w in enumerate(widths)}
        sizes = track_alterego_font_sizes(frames, size_by_width)
        self.assertEqual(sizes, {(7, "Rossi"): 17})      # median of the per-frame fits

    def test_tracks_are_independent(self):
        frames = {0: [ann(1, 60), ann(2, 30, alterego="Ben")],
                  1: [ann(1, 60), ann(2, 30, alterego="Ben")]}
        sizes = track_alterego_font_sizes(frames, size_by_width)
        self.assertEqual(sizes[(1, "Rossi")], 17)
        self.assertEqual(sizes[(2, "Ben")], 15)

    def test_ignores_hidden_untracked_and_empty(self):
        frames = {0: [ann(1, 60, to_show=False), ann(None, 60), ann(3, 60, alterego="  ")]}
        self.assertEqual(track_alterego_font_sizes(frames, size_by_width), {})

    def test_frame_sizes_constant_for_static_name(self):
        # What export does per frame: track size, then one size per name.
        widths = [(55, 46), (54, 45), (55, 46), (53, 45)]  # (Alex, Rossi) boxes
        frames = {i: [ann(1, wa, alterego="Alex"), ann(2, wr)] for i, (wa, wr) in enumerate(widths)}
        track_sizes = track_alterego_font_sizes(frames, size_by_width)

        def per_box(a):
            return track_sizes.get((a["track_id"], a["alterego"]), size_by_width(a))

        used = set()
        for anns in frames.values():
            used.update(shared_alterego_font_sizes(anns, per_box).values())
        self.assertEqual(len(used), 1, f"font size changes between frames: {used}")
        self.assertIn(used.pop(), (15, 17))              # a size some frame actually fitted

    def test_even_split_picks_a_fitted_size(self):
        frames = {i: [ann(1, w)] for i, w in enumerate([46, 45, 46, 45])}
        self.assertEqual(track_alterego_font_sizes(frames, size_by_width), {(1, "Rossi"): 15})


if __name__ == "__main__":
    unittest.main()
