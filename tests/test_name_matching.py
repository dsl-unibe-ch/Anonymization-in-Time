"""Tests for fuzzy name matching (OCR word vs dictionary name part).

OCR typos are single-letter substitutions, swaps, drops or additions. The
matcher allows: exact only if either word has up to 2 letters; one substituted
letter for 3-letter names; one edit of any kind for 4-7 letters; two for 8 or
more. Common words are never fuzzy-matched.
"""

import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Load the module file directly: importing the ait.ocr package pulls in docTR.
_spec = importlib.util.spec_from_file_location("ait_name_matching_under_test",
                                               REPO / "ait" / "ocr" / "name_matching.py")
nm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nm)


def matches(word, name_part):
    return nm._word_matches_name_part(word, name_part)[0]


def line(*words):
    boxes, x = [], 10
    for w in words:
        boxes.append({"bbox": (x, 100, x + 10 * len(w), 120), "text": w,
                      "confidence": 0.9, "line_idx": 0})
        x += 10 * len(w) + 8
    return boxes


class EditDistanceTest(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(nm._edit_distance("smith", "smith"), 0)
        self.assertEqual(nm._edit_distance("smlth", "smith"), 1)   # substitution
        self.assertEqual(nm._edit_distance("smiht", "smith"), 1)   # swapped letters
        self.assertEqual(nm._edit_distance("jon", "john"), 1)      # dropped letter
        self.assertEqual(nm._edit_distance("johnn", "john"), 1)    # extra letter
        self.assertEqual(nm._edit_distance("smtih", "smith"), 1)
        self.assertEqual(nm._edit_distance("sm", "smith"), 3)


class WordMatchTest(unittest.TestCase):
    def test_exact_and_normalised(self):
        self.assertTrue(matches("Smith", "smith"))
        self.assertTrue(matches("SMITH:", "smith"))
        self.assertTrue(matches("Müller", "muller"))

    def test_one_typo_in_mid_length_names(self):
        # previously missed: similarity (n-1)/n < 0.85 for 5-6 letters
        self.assertTrue(matches("Smlth", "smith"))
        self.assertTrue(matches("Smiht", "smith"))
        self.assertTrue(matches("Mulier", "muller"))
        self.assertTrue(matches("Allan", "allen"))

    def test_dropped_letter_in_short_names(self):
        # previously missed: position-by-position comparison counted 2 differences
        self.assertTrue(matches("Jon", "john"))
        self.assertTrue(matches("Johm", "john"))

    def test_two_typos_only_for_long_names(self):
        self.assertFalse(matches("Smlht", "smith"))          # 5 letters, 2 edits
        self.assertTrue(matches("Hendrlcs", "hendricks"))    # 9 letters, 2 edits
        self.assertFalse(matches("Hxndrlcs", "hendricks"))   # 3 edits

    def test_very_short_words_exact_only(self):
        self.assertTrue(matches("Al", "al"))
        self.assertFalse(matches("Ai", "al"))
        self.assertFalse(matches("Jo", "john"))

    def test_common_words_never_fuzzy(self):
        self.assertFalse(matches("and", "ann"))
        self.assertFalse(matches("will", "wilt"))
        self.assertTrue(matches("Will", "will"))              # exact still matches

    def test_short_names_do_not_absorb_short_words(self):
        # real false positives from the demo clip with a 30-name dictionary:
        # "Meta AI" would have blurred "AI" as the name "Ali" in every frame
        self.assertFalse(matches("AI", "ali"))
        self.assertFalse(matches("en", "ben"))
        self.assertFalse(matches("VA", "eva"))
        self.assertFalse(matches("Cali", "ali"))   # added letter on a 3-letter name
        self.assertFalse(matches("Ail", "ali"))    # swap on a 3-letter name
        self.assertTrue(matches("Tlm", "tim"))     # one substituted letter is fine

    def test_unrelated_words(self):
        self.assertFalse(matches("Lorem", "smith"))
        self.assertFalse(matches("Meta", "allen"))


class FilterByNamesTest(unittest.TestCase):
    NAMES = {"Smith John": "Alex Rossi", "Allen": "Ben Keller"}

    def shown(self, *words):
        res = nm.filter_by_names({0: line(*words)}, self.NAMES)[0]
        return [(b["text"], b.get("alterego")) for b in res if b["to_show"]]

    def test_ocr_typos_from_demo_clip(self):
        # real docTR readings from the workshop demo clip
        self.assertEqual(self.shown("Smitth", "Johm"), [("Smitth", "Alex"), ("Johm", "Rossi")])
        self.assertEqual(self.shown("Smlth", "John"), [("Smlth", "Alex"), ("John", "Rossi")])

    def test_names_in_message_text(self):
        self.assertEqual(self.shown("hello", "Allen", "how", "are", "you"), [("Allen", "Ben")])

    def test_no_false_positives_on_ordinary_text(self):
        self.assertEqual(self.shown("Lorem", "ipsum", "dolor", "sit", "amet"), [])
        self.assertEqual(self.shown("will", "you", "call", "me"), [])


if __name__ == "__main__":
    unittest.main()
