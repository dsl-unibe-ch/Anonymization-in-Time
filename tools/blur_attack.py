"""
Candidate-matching attack on the export's obfuscation (synthetic data only).

Threat model: the attacker has the exported video, the public AiT code, and a
list of candidates (e.g. the class roster, or the profile pictures of the
chat's members). For each obfuscated region they render every candidate into
the frame (the surroundings are visible, unobfuscated), run the same
obfuscation, and rank candidates by pixel error against what they observe.

Names: rows of a chat list rendered in Arial; each target name is hidden with
the method under test, the alterego is drawn on top with the export's own
function, and the frame goes through JPEG (q=85) as a stand-in for the lossy
video encode. The box-only baseline is the top-1 an attacker gets from the
OCR box alone (candidates whose tight text box matches within 1 px), which
every method leaves visible: a method that leaks nothing else scores at or
below it.

Avatars: synthetic portraits (background, face, hair, clothes, texture) in a
circular mask, matched against the full set of candidate pictures.

Usage:
    python tools/blur_attack.py [--trials 20] [--seed 0] [--methods inpaint,flat]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ait.export_video import add_custom_text_to_frame  # noqa: E402
from ait.obfuscation import obfuscate_region  # noqa: E402

FONT = "C:/Windows/Fonts/arial.ttf" if Path("C:/Windows/Fonts/arial.ttf").exists() else None

FIRST = """Anna Ben Clara David Elena Felix Gina Hugo Ines Jonas Karin Luca Mara Nico
Olga Paul Quentin Rosa Sven Tara Urs Vera Walter Xenia Yves Zoe Amir Bea Cedric Dana
Emil Fiona Gregor Hanna Ivan Julia Kevin Lea Marco Nina Oskar Petra Rafael Sara Timo
Ursula Valentin Wanda Yasmin Zora""".split()
LAST = """Meier Huber Keller Brunner Frei Weber Fischer Schmid Baumann Gerber Moser
Steiner Graf Roth Kunz Wyss Lehmann Bucher Zimmermann Marti Suter Bieri Kaufmann Hofer
Studer Widmer Berger Egli Fuchs Bosshard Rossi Bianchi Costa Ferrari Novak Horvat
Kowalski Nowak Petrovic Jovanovic Silva Santos Garcia Martin Dubois Laurent Bernard
Petit Durand Moreau""".split()
POOL = 200
ALTEREGOS = ["Alex Muster", "Sam Beispiel", "Kim Fake", "Robin Test", "Chris Demo"]
WORDS = "ok see you tomorrow lol did you get the notes yes no maybe later sure".split()


def jpeg(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- names
class NameScene:
    """One chat-list row: avatar disc, name, message preview, time."""

    def __init__(self, scale, rng):
        self.s = scale
        self.w, self.h = int(720 * scale), int(200 * scale)
        self.size = int(round(24 * scale))
        self.font = ImageFont.truetype(FONT, self.size)
        self.small = ImageFont.truetype(FONT, int(round(19 * scale)))
        self.name_xy = (int(110 * scale), int(60 * scale))
        self.msg = " ".join(rng.choice(WORDS) for _ in range(5))
        self.disc = tuple(int(c) for c in rng.integers(60, 200, 3))

    def render(self, name):
        img = Image.new("RGB", (self.w, self.h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        s = self.s
        d.ellipse([int(20 * s), int(55 * s), int(90 * s), int(125 * s)], fill=self.disc)
        d.text(self.name_xy, name, font=self.font, fill=(17, 27, 33))
        d.text((int(110 * s), int(100 * s)), self.msg, font=self.small, fill=(102, 119, 129))
        d.text((int(620 * s), int(62 * s)), "9:28 AM", font=self.small, fill=(102, 119, 129))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def box(self, name):
        """Tight text box plus 2 px, like the OCR boxes."""
        l, t, r, b = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
            self.name_xy, name, font=self.font)
        return (l - 2, t - 2, r + 2, b + 2)


def export_row(frame, box, alterego, strength, method):
    x1, y1, x2, y2 = box
    mask = np.zeros(frame.shape[:2], bool)
    if method == "fullframe":
        mask[:] = True
        return obfuscate_region(frame, mask, 3 * strength, "blur")
    patch = frame[y1:y2, x1:x2].copy()
    mask[y1:y2, x1:x2] = True
    if method == "pixelate":
        out = frame.copy()
        b = max(2, (y2 - y1) // 2)
        roi = out[y1:y2, x1:x2]
        small = cv2.resize(roi, (max(1, roi.shape[1] // b), max(1, roi.shape[0] // b)),
                           interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
    else:
        out = obfuscate_region(frame, mask, strength, method)
    return add_custom_text_to_frame(out, {"alterego": alterego}, patch, x1, y1, x2, y2)


def name_attack(method, strength, scale, trials, seed):
    rng = np.random.default_rng(seed)
    pool = sorted({f"{f} {l}" for f in FIRST for l in LAST})
    pool = list(rng.choice(pool, POOL, replace=False))
    hits1 = hits5 = 0
    width_base = 0.0
    for _ in range(trials):
        scene = NameScene(scale, rng)
        true = pool[rng.integers(len(pool))]
        alter = ALTEREGOS[rng.integers(len(ALTEREGOS))]
        box = scene.box(true)
        observed = jpeg(export_row(scene.render(true), box, alter, strength, method))
        x1, y1, x2, y2 = box
        roi = observed[y1:y2, x1:x2].astype(np.float32) if method != "fullframe" \
            else observed.astype(np.float32)
        same_w = 0
        errs = []
        for cand in pool:
            if max(abs(a - c) for a, c in zip(scene.box(cand), box)) <= 1:
                same_w += 1
            sim = export_row(scene.render(cand), box, alter, strength, method)
            sim = sim[y1:y2, x1:x2] if method != "fullframe" else sim
            errs.append(float(np.mean((sim.astype(np.float32) - roi) ** 2)))
        i = pool.index(true)
        hits1 += expected_hit(errs, i, 1)
        hits5 += expected_hit(errs, i, 5)
        width_base += 1.0 / max(1, same_w)
    return hits1 / trials, hits5 / trials, width_base / trials, len(pool)


def expected_hit(errs, true_idx, k):
    """P(true candidate is in the attacker's top k), ties broken at random.

    Candidates the obfuscation makes indistinguishable produce identical
    errors; counting them by list order would credit the attacker with luck.
    """
    errs = np.asarray(errs, dtype=np.float64)
    e = errs[true_idx]
    tol = 1e-6 * max(1.0, abs(e))
    better = int(np.sum(errs < e - tol))
    tied = int(np.sum(np.abs(errs - e) <= tol))
    return float(np.clip((k - better) / tied, 0.0, 1.0))


# ---------------------------------------------------------------- avatars
def make_avatar(rng, d):
    img = Image.new("RGB", (d, d))
    dr = ImageDraw.Draw(img)
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    for y in range(d):
        t = y / d
        dr.line([(0, y), (d, y)], fill=tuple(int(v) for v in (1 - t) * c1 + t * c2))
    cx = d * rng.uniform(0.35, 0.65)
    cy = d * rng.uniform(0.35, 0.5)
    r = d * rng.uniform(0.16, 0.24)
    body = tuple(int(v) for v in rng.integers(0, 256, 3))
    dr.ellipse([cx - 1.8 * r, cy + 1.2 * r, cx + 1.8 * r, cy + 4 * r], fill=body)
    skin = [(255, 220, 190), (230, 180, 140), (190, 140, 100), (140, 95, 60), (90, 60, 40)]
    dr.ellipse([cx - r, cy - 1.2 * r, cx + r, cy + 1.2 * r], fill=skin[rng.integers(5)])
    hair = tuple(int(v) for v in rng.integers(0, 120, 3))
    dr.chord([cx - 1.1 * r, cy - 1.45 * r, cx + 1.1 * r, cy + 0.4 * r], 180, 360, fill=hair)
    a = np.array(img).astype(np.int16) + rng.integers(-12, 13, (d, d, 3))
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def avatar_attack(method, strength, d, n, trials, seed):
    rng = np.random.default_rng(seed)
    avatars = [make_avatar(rng, d) for _ in range(n)]
    pad = 2 * d
    H, W = d + 2 * pad, d + 2 * pad
    yy, xx = np.mgrid[:H, :W]
    mask = (yy - H / 2 + 0.5) ** 2 + (xx - W / 2 + 0.5) ** 2 <= (d / 2) ** 2

    def place(av):
        f = np.full((H, W, 3), 255, np.uint8)
        f[pad:pad + d, pad:pad + d][mask[pad:pad + d, pad:pad + d]] = \
            av[mask[pad:pad + d, pad:pad + d]]
        if method == "fullframe":
            return obfuscate_region(f, np.ones((H, W), bool), 3 * strength, "blur")
        if method == "pixelate":
            b = max(2, d // 6)
            roi = f[pad:pad + d, pad:pad + d]
            sm = cv2.resize(roi, (d // b, d // b), interpolation=cv2.INTER_AREA)
            out = f.copy()
            pix = cv2.resize(sm, (d, d), interpolation=cv2.INTER_NEAREST)
            out[pad:pad + d, pad:pad + d][mask[pad:pad + d, pad:pad + d]] = \
                pix[mask[pad:pad + d, pad:pad + d]]
            return out
        return obfuscate_region(f, mask, strength, method)

    sims = np.stack([place(av)[mask].astype(np.float32) for av in avatars])
    hits = 0
    for _ in range(trials):
        i = int(rng.integers(n))
        obs = jpeg(place(avatars[i]))[mask].astype(np.float32)
        hits += expected_hit(((sims - obs) ** 2).mean(axis=(1, 2)), i, 1)
    return hits / trials


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="blur,pixelate,fullframe,flat,inpaint",
                    help="comma-separated subset to run")
    ap.add_argument("--no-avatars", action="store_true")
    args = ap.parse_args()

    wanted = set(args.methods.split(","))
    configs = [(m, k) for m, k in
               [("blur", 31), ("blur", 51), ("blur", 91), ("blur", 151),
                ("pixelate", 0), ("fullframe", 91), ("flat", 91), ("inpaint", 91)]
               if m in wanted]
    label = {"blur": "gaussian blur k={k}", "pixelate": "pixelate",
             "fullframe": "full-frame blur k={k3}", "flat": "flat gray + blur k={k}",
             "inpaint": "inpaint + blur k={k}"}

    print(f"NAMES: {POOL} candidates; top-1 / top-5 vs box-only baseline")
    for scale, desc in ((0.5, "small text ~12 px (360p-like)"),
                        (1.0, "text ~24 px (720p)"),
                        (1.5, "text ~36 px (1080p)")):
        print(f"  {desc}")
        for method, k in configs:
            n = args.trials if method != "fullframe" else max(1, args.trials // 3)
            t1, t5, wb, _ = name_attack(method, k, scale, n, args.seed)
            name = label[method].format(k=k, k3=3 * k)
            print(f"    {name:26s} top-1 {t1:6.1%}  top-5 {t5:6.1%}   box-only {wb:6.1%}")

    if args.no_avatars:
        return
    print("\nAVATARS: 200 candidate pictures; top-1 (chance 0.5%)")
    for d, desc in ((64, "64 px avatar (720p)"), (140, "140 px avatar (1080p)")):
        print(f"  {desc}")
        for method, k in configs:
            acc = avatar_attack(method, k, d, 200, args.trials, args.seed)
            name = label[method].format(k=k, k3=3 * k)
            print(f"    {name:26s} top-1 {acc:6.1%}")


if __name__ == "__main__":
    main()
