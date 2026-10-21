"""Tests for carrying names on while they scroll out of / into view, and for
track assignment during fast scrolls.

Synthetic chat: a header label that never moves, and a list scrolling up
10 px per frame with a name ("Amina", shown) and message text (hidden).
"""

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Load the modules directly: importing the ait.ocr package pulls in docTR.
_spec = importlib.util.spec_from_file_location("ait_scroll_extension_under_test",
                                               REPO / "ait" / "ocr" / "scroll_extension.py")
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)

if "ait.ocr" not in sys.modules:
    _pkg = types.ModuleType("ait.ocr")
    _pkg.__path__ = [str(REPO / "ait" / "ocr")]
    sys.modules["ait.ocr"] = _pkg
stab = importlib.import_module("ait.ocr.stabilization")

FRAME_SIZE = (300, 600)  # width, height


def box(track_id, text, y, x=20, h=20, w=60, to_show=False):
    return {"bbox": (x, y, x + w, y + h), "parent_box": (x, y, x + w, y + h),
            "text": text, "track_id": track_id, "to_show": to_show}


def chat(name_frames, n_frames=10, speed=-10, name_y0=100):
    """Name seen only in name_frames; message and header in every frame."""
    frames = {}
    for f in range(n_frames):
        boxes = [box(0, "Chats", 5),                                  # static header
                 box(2, "hello", 300 + speed * f)]                    # scrolling message
        if f in name_frames:
            boxes.append(box(1, "Amina", name_y0 + speed * f, to_show=True))
        frames[f] = boxes
    return frames


def extended(frames):
    return {f: [b["bbox"] for b in boxes if b.get("extrapolated")]
            for f, boxes in frames.items() if any(b.get("extrapolated") for b in boxes)}


class ScrollOutTest(unittest.TestCase):
    def test_name_lost_while_scrolling_is_carried_on(self):
        out = se.extend_scrolling_names(chat(range(0, 5)), FRAME_SIZE)
        # last seen at y=60 (frame 4); carried on with the list until it has
        # moved its height plus one step past that
        self.assertEqual(extended(out), {5: [(20, 50, 80, 70)], 6: [(20, 40, 80, 60)]})
        added = out[5][-1]
        self.assertEqual(added["track_id"], 1)       # viewer toggles it with the name
        self.assertTrue(added["to_show"])
        self.assertEqual(added["parent_box"], (20, 50, 80, 70))

    def test_name_scrolling_in_is_extended_backwards(self):
        out = se.extend_scrolling_names(chat(range(5, 10), speed=-10, name_y0=600), FRAME_SIZE)
        # first seen at y=550 (frame 5): earlier frames get the box lower down,
        # until it is below the frame (y >= 600)
        self.assertEqual(extended(out), {4: [(20, 560, 80, 580)], 3: [(20, 570, 80, 590)]})

    def test_follows_the_scroll_as_it_slows_down(self):
        frames = chat(range(0, 5))
        for f, dy in [(5, -6), (6, -3), (7, -1)]:   # list decelerates after frame 4
            msg = frames[f][1]
            y = frames[f - 1][1]["bbox"][1] + dy
            frames[f][1] = {**msg, "bbox": (20, y, 80, y + 20)}
        out = se.extend_scrolling_names(frames, FRAME_SIZE, max_frames=3)
        self.assertEqual([b[1] for f in (5, 6, 7) for b in extended(out)[f]], [54, 51, 50])

    def test_static_name_that_disappears_is_not_extended(self):
        out = se.extend_scrolling_names(chat(range(0, 5), speed=0), FRAME_SIZE)
        self.assertEqual(extended(out), {})

    def test_hidden_text_is_not_extended(self):
        frames = chat(range(0, 5))
        for f in range(5):
            frames[f][2]["to_show"] = False
        self.assertEqual(extended(se.extend_scrolling_names(frames, FRAME_SIZE)), {})

    def test_stops_at_frame_edge(self):
        out = se.extend_scrolling_names(chat(range(0, 9), name_y0=95), FRAME_SIZE)
        # last seen at y=15 (frame 8); frame 9 would put it at y=5 (still
        # partly inside the frame)
        self.assertEqual(extended(out), {9: [(20, 5, 80, 25)]})
        out = se.extend_scrolling_names(chat(range(0, 9), name_y0=78, speed=-10), FRAME_SIZE)
        self.assertEqual(extended(out), {9: [(20, -12, 80, 8)]})   # runs past the edge, kept full size

    def test_word_unmatched_at_the_edge_stays_shown(self):
        # first name cut off and misread: the surname track stops matching
        frames = chat(range(0, 6))
        for f in range(0, 6):
            frames[f][2]["name"], frames[f][2]["alterego"] = "Amina", "Lejla"
        frames[5][2] = {**frames[5][2], "to_show": False}
        del frames[5][2]["name"], frames[5][2]["alterego"]
        out = se.extend_scrolling_names(frames, FRAME_SIZE)
        self.assertTrue(out[5][2]["to_show"])
        self.assertEqual(out[5][2]["alterego"], "Lejla")
        self.assertFalse(out[5][1]["to_show"])            # other text untouched
        self.assertEqual(extended(out), {6: [(20, 40, 80, 60)], 7: [(20, 30, 80, 50)]})

    def test_name_seen_once_uses_the_scroll_around_it(self):
        frames = chat(range(5, 10), name_y0=600)
        for f in range(10):  # more moving text, so the scroll is measurable
            frames[f] += [box(3, "how", 350 - 10 * f), box(4, "are", 400 - 10 * f)]
        frames[5][2] = {**frames[5][2], "track_id": 9}    # misread: its own one-frame track
        out = se.extend_scrolling_names(frames, FRAME_SIZE)
        self.assertEqual(extended(out)[4], [(20, 560, 80, 580)])

    def test_stops_when_ocr_reads_the_name_again(self):
        frames = chat(range(0, 5))
        frames[5].append(box(7, "Amina", 50, to_show=True))  # new track, same place
        self.assertEqual(extended(se.extend_scrolling_names(frames, FRAME_SIZE)), {})


class FastScrollTrackingTest(unittest.TestCase):
    def test_fast_scroll_keeps_track_ids(self):
        # 120 px per frame: beyond max_dist, but the scroll shift is measured
        texts = ["Amina", "Lejla", "Marko", "hello", "there"]
        frames = {}
        for f in range(3):
            boxes = [{"bbox": (20, 5, 80, 25), "text": "Chats"}]    # static header
            for i, t in enumerate(texts):
                y = 100 + 60 * i - 120 * f
                boxes.append({"bbox": (20, y, 80, y + 20), "text": t})
            frames[f] = boxes
        tracked = stab._assign_tracks(frames)
        for i in range(len(texts) + 1):
            ids = {tracked[f][i]["track_id"] for f in range(3)}
            self.assertEqual(len(ids), 1, f"track split for {tracked[0][i]['text']}")

    def test_static_text_unaffected(self):
        frames = {f: [{"bbox": (20, 5, 80, 25), "text": "Chats"}] for f in range(3)}
        tracked = stab._assign_tracks(frames)
        self.assertEqual({tracked[f][0]["track_id"] for f in range(3)}, {0})


if __name__ == "__main__":
    unittest.main()
