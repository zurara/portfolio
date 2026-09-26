"""Repaint an image as an impressionist oil sketch with short brush strokes.

Layered stroke-based rendering after Hertzmann (1998): big brushes lay in the
masses, smaller brushes only go where the canvas still differs from the
reference. Strokes follow the edges of the reference (structure tensor), and
each stroke is drawn as a bundle of bristle lines with slightly broken color.

Usage: python impressionist.py [--src folk_winter.png] [--out impressionist.png]
"""
import argparse
import colorsys
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

BRUSHES = [22, 12, 7, 4]          # brush radius in px, coarse to fine
THRESHOLD = [0, 18, 16, 14]       # min color error before a brush paints (0 = paint everywhere)
GROUND = (222, 208, 184)          # warm primed canvas


# ---------------------------------------------------------------- color grading

def monet_grade(img, seed):
    """Push the flat folk palette toward plein-air color: warm light,
    blue-violet shadows, no pure black, and colour variation in the snow."""
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32) / 255
    h, w, _ = a.shape
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)

    # shadows lean violet-blue, lights lean cream
    shadow = np.array([0.33, 0.30, 0.46], np.float32)
    light = np.array([1.0, 0.95, 0.84], np.float32)
    ks = np.clip((0.45 - lum) / 0.45, 0, 1)[..., None] ** 0.8
    kl = np.clip((lum - 0.7) / 0.3, 0, 1)[..., None]
    a = a * (1 - 0.55 * ks) + shadow * 0.55 * ks
    a = a * (1 - 0.5 * kl) + light * 0.5 * kl

    # large soft patches of lilac / blue / peach across the snow
    def blob(scale):
        n = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
        return ndimage.zoom(n, scale, order=3)[:h, :w]
    snow = np.clip((lum - 0.8) / 0.2, 0, 1)[..., None]
    cool = np.clip(blob(90) * 1.4 - 0.5, 0, 1)[..., None]
    warm = np.clip(blob(70) * 1.4 - 0.6, 0, 1)[..., None]
    lilac = np.array([0.72, 0.72, 0.88], np.float32)
    peach = np.array([1.0, 0.86, 0.74], np.float32)
    a = a * (1 - snow * cool * 0.75) + lilac * snow * cool * 0.75
    a = a * (1 - snow * warm * 0.5) + peach * snow * warm * 0.5

    # sky: overcast with rose near the horizon and green-grey above
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    # sky pixels only: above the horizon and not a dark trunk or branch
    sky = (np.arange(h) < 385)[:, None, None] * np.clip((lum[..., None] - 0.22) / 0.1, 0, 1)
    rose = np.array([0.80, 0.70, 0.70], np.float32)
    k = sky * np.clip((yy - 0.15) / 0.22, 0, 1) * 0.35
    a = a * (1 - k) + rose * k
    haze = np.array([0.66, 0.70, 0.66], np.float32)
    ochre = np.array([0.82, 0.76, 0.60], np.float32)
    a = a * (1 - sky * 0.28) + haze * sky * 0.28
    ks = sky * np.clip(blob(110) * 1.5 - 0.7, 0, 1)[..., None] * 0.35
    a = a * (1 - ks) + ochre * ks
    kr = sky * np.clip(blob(80) * 1.5 - 0.7, 0, 1)[..., None] * 0.3
    a = a * (1 - kr) + rose * kr

    # ice: mix in blues and pale yellow-greens so it is not one flat teal
    ice = ((a[..., 1] - a[..., 0]) > 0.12)[..., None] * (lum > 0.45)[..., None]
    kb = ice * np.clip(blob(40) * 1.4 - 0.4, 0, 1)[..., None] * 0.45
    a = a * (1 - kb) + np.array([0.55, 0.68, 0.80], np.float32) * kb

    # lift contrast floor: nothing darker than a deep blue-brown
    a = np.maximum(a, np.array([0.16, 0.13, 0.20], np.float32))
    return np.clip(a * 255, 0, 255)


# ---------------------------------------------------------------- orientation

def orientation_field(ref, sigma):
    """Angle of the edge (not the gradient) at each pixel, from a smoothed
    structure tensor. Flat regions get a gentle horizontal drift."""
    lum = ref @ np.array([0.299, 0.587, 0.114], np.float32)
    gx = ndimage.sobel(lum, axis=1)
    gy = ndimage.sobel(lum, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    grad_angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    strength = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return grad_angle + math.pi / 2, strength / (strength.max() + 1e-6)


# ---------------------------------------------------------------- strokes

def broken_color(rgb, rng_hue=0.05, rng_val=0.08):
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + random.uniform(-rng_hue, rng_hue)) % 1
    s = min(1, max(0, s * random.uniform(0.9, 1.35) + random.uniform(0, 0.04)))
    v = min(1, max(0, v + random.uniform(-rng_val, rng_val)))
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))


def paint_stroke(draw, x, y, angle, r, color, alpha):
    """One brush stroke = a bundle of parallel bristle lines."""
    length = r * random.uniform(3.0, 5.5)
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx
    bristles = max(3, int(r * 1.3))
    width = max(1, int(r * 2.2 / bristles) + 1)
    for i in range(bristles):
        off = (i / (bristles - 1) - 0.5) * 2 * r * random.uniform(0.9, 1.05)
        # bristles at the edge are shorter, like a real flat brush lifting off
        edge = 1 - abs(i / (bristles - 1) - 0.5) * 2
        ln = length * (0.65 + 0.35 * edge) * random.uniform(0.85, 1.1)
        start = random.uniform(-0.1, 0.1) * length
        cx, cy = x + nx * off + dx * start, y + ny * off + dy * start
        c = tuple(max(0, min(255, v + random.randint(-9, 9))) for v in color)
        draw.line([(cx - dx * ln / 2, cy - dy * ln / 2), (cx + dx * ln / 2, cy + dy * ln / 2)],
                  fill=c + (int(alpha * random.uniform(0.7, 1.0)),), width=width)


def paint(ref, seed):
    random.seed(seed)
    h, w, _ = ref.shape
    canvas = Image.new("RGB", (w, h), GROUND)
    draw = ImageDraw.Draw(canvas, "RGBA")
    total = 0
    for layer, r in enumerate(BRUSHES):
        blurred = ndimage.gaussian_filter(ref, sigma=(r * 0.6, r * 0.6, 0))
        angle, strength = orientation_field(blurred, sigma=r * 1.5)
        cur = np.asarray(canvas).astype(np.float32)
        err = np.sqrt(((cur - blurred) ** 2).sum(axis=2))
        grid = max(2, int(r * 1.1))
        strokes = []
        for gy in range(0, h, grid):
            for gx in range(0, w, grid):
                cell = err[gy:gy + grid, gx:gx + grid]
                if cell.mean() <= THRESHOLD[layer]:
                    continue
                # paint where the cell is most wrong, as Hertzmann does
                iy, ix = np.unravel_index(np.argmax(cell), cell.shape)
                y = min(h - 1, gy + iy + random.randint(-1, 1))
                x = min(w - 1, gx + ix + random.randint(-1, 1))
                strokes.append((x, y))
        random.shuffle(strokes)
        for x, y in strokes:
            a = angle[y, x]
            # weak edges: let the brush wander, mostly horizontal in the snow
            if strength[y, x] < 0.02:
                a = random.gauss(0, 0.5)
            a += random.gauss(0, 0.18)
            col = broken_color(tuple(int(v) for v in blurred[y, x]))
            alpha = 235 if layer else 255
            paint_stroke(draw, x, y, a, r * random.uniform(0.8, 1.1), col, alpha)
        total += len(strokes)
        print(f"layer r={r:>2}: {len(strokes)} strokes")
    print("total strokes:", total)
    return canvas


# ---------------------------------------------------------------- finish

def impasto(canvas):
    """Fake paint relief: light the luminance detail from the upper left."""
    a = np.asarray(canvas).astype(np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    detail = lum - ndimage.gaussian_filter(lum, 2)
    relief = np.zeros_like(detail)
    relief[1:, 1:] = detail[:-1, :-1] - detail[1:, 1:]
    weave = np.random.normal(0, 3, lum.shape)
    a += (relief * 0.8 + weave)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="folk_winter.png")
    ap.add_argument("--out", default="impressionist.png")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    src = Image.open(a.src).convert("RGB")
    ref = monet_grade(src, a.seed)
    # paint past the borders so edge strokes are as dense as the middle
    m = 30
    ref = np.pad(ref, ((m, m), (m, m), (0, 0)), mode="reflect")
    painted = paint(ref, a.seed).crop((m, m, m + src.width, m + src.height))
    impasto(painted).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
