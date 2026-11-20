"""Tests for ait.obfuscation: the hidden content must not reach the output."""

import unittest

import cv2
import numpy as np

from ait.obfuscation import infer_text_color, obfuscate_region


def _scene(seed):
    rng = np.random.default_rng(seed)
    img = np.full((120, 200, 3), 240, np.uint8)
    img[:, :40] = (30, 120, 200)          # coloured context left of the box
    img[90:] = rng.integers(0, 256, (30, 200, 3), dtype=np.uint8)
    return img


def _mask():
    m = np.zeros((120, 200), bool)
    m[40:70, 50:150] = True               # a name box
    yy, xx = np.mgrid[:120, :200]
    m |= (yy - 30) ** 2 + (xx - 175) ** 2 <= 15 ** 2   # an avatar disc
    return m


class ObfuscationTest(unittest.TestCase):
    def _two_secrets(self):
        a, b = _scene(0), _scene(0)
        rng = np.random.default_rng(1)
        m = _mask()
        a[m] = rng.integers(0, 256, (m.sum(), 3), dtype=np.uint8)
        b[m] = 255 - a[m]
        return a, b, m

    def test_safe_methods_do_not_depend_on_hidden_pixels(self):
        a, b, m = self._two_secrets()
        for method in ("inpaint", "flat"):
            with self.subTest(method=method):
                np.testing.assert_array_equal(
                    obfuscate_region(a, m, 31, method), obfuscate_region(b, m, 31, method))

    def test_legacy_blur_leaks_hidden_pixels(self):
        a, b, m = self._two_secrets()
        self.assertFalse(np.array_equal(
            obfuscate_region(a, m, 31, "blur"), obfuscate_region(b, m, 31, "blur")))

    def test_pixels_outside_mask_unchanged(self):
        img, m = _scene(2), _mask()
        for method in ("inpaint", "flat", "blur"):
            out = obfuscate_region(img, m, 51, method)
            np.testing.assert_array_equal(out[~m], img[~m])

    def test_legacy_blur_matches_full_frame_gaussian(self):
        img, m = _scene(3), _mask()
        full = cv2.GaussianBlur(img, (51, 51), 0)
        np.testing.assert_array_equal(obfuscate_region(img, m, 51, "blur")[m], full[m])

    def test_even_strength_and_empty_mask(self):
        img = _scene(4)
        out = obfuscate_region(img, np.zeros(img.shape[:2], bool), 90)
        np.testing.assert_array_equal(out, img)
        self.assertIsNot(out, img)
        obfuscate_region(img, _mask(), 90)   # even kernel is made odd

    def test_full_frame_mask_does_not_crash(self):
        img = _scene(5)
        out = obfuscate_region(img, np.ones(img.shape[:2], bool), 31, "inpaint")
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(len(np.unique(out.reshape(-1, 3), axis=0)), 1)

    def test_text_color_does_not_depend_on_the_letters(self):
        colors = set()
        for text in ("Il Li", "Mara Moser", "Wendy Bwmm", "Jo Ux"):
            patch = np.full((30, 160, 3), (250, 245, 240), np.uint8)
            cv2.putText(patch, text, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (33, 27, 17), 2, cv2.LINE_AA)
            colors.add(infer_text_color(patch))
        self.assertEqual(colors, {(32, 32, 32)})

    def test_text_color_fallback(self):
        self.assertEqual(infer_text_color(None, (1, 2, 3)), (1, 2, 3))
        flat = np.full((10, 10, 3), 128, np.uint8)
        self.assertEqual(infer_text_color(flat, (1, 2, 3)), (1, 2, 3))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            obfuscate_region(_scene(0), _mask(), 31, "swirl")


if __name__ == "__main__":
    unittest.main()
