"""Paint a photo as an alla-prima impressionist oil sketch.

Compared with impressionist.py this one tries to look *painted*:
  - strokes are curved and follow the form, and stop when the colour under
    the brush changes, so one stroke covers one patch of colour
  - every stroke is a loaded flat brush: bristles carry slightly different
    paint, the tail runs dry and breaks up, the warm ground shows through
  - each region has its own handling (long flat strokes in fields and sky,
    curling strokes in clouds, short vertical dabs in the tree line)
  - the barn and the red roof are not auto-painted: they are laid in with a
    dozen hand-placed single strokes, the way a painter would state them
  - a height map records the paint ridges and is lit for impasto

Usage: python paint_photo.py [--src field.jpg] [--out field_painted.png]
"""
import argparse
import colorsys
import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SCALE = 0.75                       # photo px -> canvas px
BRUSHES = [34, 20, 11, 6]          # brush radius (canvas px), coarse to fine
THRESHOLD = [0, 18, 20, 24]        # colour error a layer needs before it paints
GROUND = (196, 146, 104)           # warm sienna imprimatura
LUM = np.array([0.299, 0.587, 0.114], np.float32)

# region bands, in photo coordinates (the photo is 1932 x 2576)
SKY_TOP_END = 590
CLOUDS_END = 1400
TREES_TOP = 1880
FIELD_TOP = 2076
CORN_END = 2210
GRASS_LIGHT_END = 2310


# ---------------------------------------------------------------- colour

def grade(a, seed):
    """Photo -> painter's colour: fuller chroma, violet shadows, warm lights,
    and a slow drift of hue across big areas (broken colour at large scale)."""
    rng = np.random.default_rng(seed)
    h, w, _ = a.shape
    a = a / 255
    lum = (a @ LUM)[..., None]
    a = np.clip(lum + (a - lum) * 1.3, 0, 1)                       # chroma
    ks = np.clip((0.38 - lum) / 0.38, 0, 1) * 0.45
    a = a * (1 - ks) + np.array([0.20, 0.22, 0.36], np.float32) * ks  # violet shadows
    kl = np.clip((lum - 0.72) / 0.28, 0, 1) * 0.35
    a = a * (1 - kl) + np.array([1.0, 0.93, 0.82], np.float32) * kl   # warm lights

    def blob(scale):
        n = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
        return ndimage.zoom(n, scale, order=3)[:h, :w, None]
    for col, scale, amt in (((0.95, 0.72, 0.70), 120, 0.14),   # rose
                            ((0.62, 0.70, 0.92), 150, 0.14),   # blue
                            ((0.95, 0.85, 0.55), 100, 0.10)):  # ochre
        k = np.clip(blob(scale) * 1.6 - 0.8, 0, 1) * amt
        a = a * (1 - k) + np.array(col, np.float32) * k
    return np.clip(a * 255, 0, 255)


def broken_color(rgb, hue=0.03, val=0.07, sat=(0.95, 1.3)):
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + random.uniform(-hue, hue)) % 1
    s = min(1, s * random.uniform(*sat))
    v = min(1, max(0, v + random.uniform(-val, val)))
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))


def mix(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


# ---------------------------------------------------------------- brush

class Canvas:
    def __init__(self, w, h):
        self.img = Image.new("RGB", (w, h), GROUND)
        self.height = Image.new("L", (w, h), 0)
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self.dh = ImageDraw.Draw(self.height)

    def stroke(self, pts, r, color, color2=None, alpha=235, dry=0.35, wobble=0.0):
        """Flat loaded brush along a polyline.
        color2: a second paint picked up on part of the bristles (dirty brush).
        dry: how much of the tail can break up into separate bristles."""
        pts = resample(pts, max(1.5, r * 0.35))
        if len(pts) < 3:
            return
        normals = []
        for i in range(len(pts)):
            a = pts[max(0, i - 1)]
            b = pts[min(len(pts) - 1, i + 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) or 1
            normals.append((-dy / L, dx / L))
        N = len(pts)
        n = max(3, min(28, int(r * 0.9)))
        bw = max(1, int(2 * r / n + 1.5))
        tail_col = mix(color, color2, 0.3) if color2 is not None else color
        for i in range(n):
            u = abs(i / (n - 1) - 0.5) * 2              # 0 in the middle, 1 at the edge
            off = (i / (n - 1) - 0.5) * 2 * r * random.uniform(0.9, 1.05)
            c = color
            if color2 is not None and random.random() < 0.3:
                c = mix(color, color2, random.uniform(0.3, 0.8))
            c = tuple(max(0, min(255, v + random.randint(-7, 7))) for v in c)
            # rounded head: edge bristles touch down a little later
            s0 = int(N * (0.2 * u ** 1.6 * random.random()))
            # tail: bristles run dry at different points, edges first
            cut = dry * random.random() ** 1.4 * (0.3 + u) + 0.12 * u ** 2
            s1 = max(s0 + 2, int(N * (1 - min(0.85, cut))))
            jit = wobble * r
            line = [(p[0] + nm[0] * off + random.uniform(-jit, jit),
                     p[1] + nm[1] * off + random.uniform(-jit, jit))
                    for p, nm in zip(pts[s0:s1], normals[s0:s1])]
            if len(line) < 2:
                continue
            a = int(alpha * random.uniform(0.8, 1.0))
            head = max(2, int(len(line) * 0.65))
            self.d.line(line[:head], fill=c + (a,), width=bw, joint="curve")
            ridge = int(120 + 50 * u + random.randint(-15, 15))
            self.dh.line(line[:head], fill=ridge, width=bw)
            if len(line) > head and random.random() < 0.8:
                tc = mix(c, tail_col, 0.6)
                self.d.line(line[head - 1:], fill=tc + (int(a * 0.6),), width=max(1, bw - 1))
                self.dh.line(line[head - 1:], fill=max(0, ridge - 50), width=max(1, bw - 1))

    def flick(self, pts, width, color, alpha=230):
        """A thin single-hair touch (grass, corn leaves)."""
        self.d.line(pts, fill=color + (alpha,), width=width, joint="curve")
        self.dh.line(pts, fill=200, width=width)


def resample(pts, ds):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        L = math.dist(a, b)
        k = max(1, int(L / ds))
        for j in range(1, k + 1):
            t = j / k
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


# ---------------------------------------------------------------- orientation

def structure_angle(ref, sigma):
    lum = ref @ LUM
    gx = ndimage.sobel(lum, axis=1)
    gy = ndimage.sobel(lum, axis=0)
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
    strength = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return ang, strength / (np.percentile(strength, 99) + 1e-6)


def region(x, y, tree_mask):
    """Brush handling per part of the picture (y in photo coordinates).
    Returns (mode, length multiplier, allowed curvature)."""
    if TREES_TOP <= y < FIELD_TOP + 4 and tree_mask:
        return "dab", 0.45, 0.0
    if y < SKY_TOP_END:
        return "flat", 1.4, 0.4
    if y < CLOUDS_END:
        return "follow", 1.0, 1.0
    if y < FIELD_TOP:
        return "flat", 1.8, 0.15
    if y < CORN_END:
        return "flat", 1.6, 0.05
    return "flat", 3.0, 0.1


# ---------------------------------------------------------------- auto layers

def paint_layers(cv, ref, barn_mask, trees, seed):
    random.seed(seed)
    h, w, _ = ref.shape
    for layer, r in enumerate(BRUSHES):
        blurred = ndimage.gaussian_filter(ref, sigma=(r * 0.5, r * 0.5, 0))
        ang, strength = structure_angle(blurred, r * 1.2)
        cur = np.asarray(cv.img).astype(np.float32)
        err = np.sqrt(((cur - blurred) ** 2).sum(axis=2))
        if layer >= 1:
            err[barn_mask] = 0            # the barn gets its own strokes
        grid = max(3, int(r * 1.0))
        seeds = []
        for gy in range(0, h, grid):
            for gx in range(0, w, grid):
                cell = err[gy:gy + grid, gx:gx + grid]
                if cell.mean() <= THRESHOLD[layer]:
                    continue
                iy, ix = np.unravel_index(np.argmax(cell), cell.shape)
                seeds.append((gx + ix, gy + iy))
        random.shuffle(seeds)
        for x, y in seeds:
            py = y / SCALE
            mode, lmul, curl = region(x, py, trees[y, x])
            rr = r * random.uniform(0.85, 1.1)
            if mode == "dab":
                rr = min(rr, 9)
            base = tuple(float(v) for v in blurred[y, x])
            color = broken_color(base)
            # direction
            if mode == "dab":
                a0 = -math.pi / 2 + random.gauss(0, 0.5)
            elif mode == "flat":
                spread = 0.35 if py < SKY_TOP_END else 0.12
                a0 = random.gauss(0, spread) + (random.random() < 0.5) * math.pi
            else:
                a0 = ang[y, x] if strength[y, x] > 0.05 else random.gauss(0, 0.4)
            # grow the stroke until the colour under it changes
            pts = [(x, y)]
            step = rr * 0.8
            max_steps = max(2, int(lmul * random.uniform(3, 6)))
            a = a0
            for _ in range(max_steps):
                cx, cy = pts[-1]
                ix, iy = int(cx), int(cy)
                if not (0 <= ix < w and 0 <= iy < h):
                    break
                if len(pts) > 2 and math.dist(blurred[iy, ix], base) > 38:
                    break
                if curl > 0 and strength[iy, ix] > 0.05:
                    t = ang[iy, ix]
                    if math.cos(t - a) < 0:
                        t += math.pi
                    a = a + (math.atan2(math.sin(t - a), math.cos(t - a))) * curl * 0.5
                a += random.gauss(0, 0.04)
                pts.append((cx + math.cos(a) * step, cy + math.sin(a) * step))
            # dirty brush: a little of the neighbouring colour
            nx = int(min(w - 1, max(0, x + math.sin(a0) * rr * 2)))
            ny = int(min(h - 1, max(0, y - math.cos(a0) * rr * 2)))
            color2 = broken_color(tuple(float(v) for v in blurred[ny, nx]))
            cv.stroke(pts, rr, color, color2, alpha=250 if layer == 0 else 238,
                      dry=0.25 if layer == 0 else 0.45)
        print(f"layer r={r:>2}: {len(seeds)} strokes")


# ---------------------------------------------------------------- barn, stated in single strokes

def barn_strokes(cv):
    """Photo-coordinate strokes for the barn and the red roof. Each call is
    one stroke of a loaded brush; no retouching."""
    S = SCALE

    def st(pts, r, col, col2=None, dry=0.3, alpha=245):
        cv.stroke([(x * S, y * S) for x, y in pts], r * S, broken_color(col, 0.01, 0.03, (1, 1.1)),
                  col2, alpha=alpha, dry=dry, wobble=0.04)

    def line(p0, p1, n=8):
        return [(p0[0] + (p1[0] - p0[0]) * t / n, p0[1] + (p1[1] - p0[1]) * t / n) for t in range(n + 1)]

    apex = (1200, 1989)

    def gable_x(y):
        """Left and right edge of the gable wall at height y."""
        left = max(1057, apex[0] - (y - apex[1]) * 195 / 60)
        right = min(1352, apex[0] + (y - apex[1]) * 152 / 57)
        return left, right
    # gable wall: four broad horizontal strokes, the top one short under the apex
    for yy, r, col in ((1999, 6, (196, 186, 176)), (2012, 11, (190, 180, 172)), (2032, 12, (178, 168, 160)),
                       (2050, 12, (166, 158, 154)), (2067, 11, (150, 144, 146))):
        x0, x1 = gable_x(yy - r * 0.6)
        if random.random() < 0.5:
            x0, x1 = x1, x0
        st(line((x0, yy), (x1, yy + random.uniform(-2, 2)), 10), r, col, (140, 134, 140), dry=0.18)
    # lean-to on the left, in shadow
    st(line((1004, 2062), (1062, 2064), 4), 13, (62, 66, 90), (88, 84, 96), dry=0.1)
    # long roof plane: three horizontal strokes, lighter at the top
    for yy, col in ((2000, (178, 190, 208)), (2016, (156, 170, 192)), (2032, (136, 148, 170))):
        x0 = apex[0] + (yy - apex[1]) * (150 / 56) - 6
        st(line((x0, yy), (1562, yy + 3), 10), 9, col, (196, 204, 214), dry=0.3)
        st(line((1564, yy + 3), (x0 + 60, yy + 1), 10), 8, col, (196, 204, 214), dry=0.45, alpha=200)
    # lit side wall, warm
    st(line((1352, 2054), (1558, 2056), 10), 8, (232, 212, 170), (246, 232, 196), dry=0.3)
    st(line((1352, 2068), (1558, 2069), 10), 8, (208, 186, 150), (180, 160, 140), dry=0.4)
    # the drawing: dark fascia strokes along both gable edges and the eave
    st(line(apex, (1002, 2054), 10), 3.6, (86, 76, 74), dry=0.15)
    st(line(apex, (1352, 2046), 8), 3.2, (86, 76, 74), dry=0.15)
    st(line((1352, 2046), (1560, 2047), 10), 2.4, (96, 90, 96), dry=0.4, alpha=220)
    # red roof on the right
    st(line((1706, 2046), (1932, 2044), 10), 5, (170, 96, 104), dry=0.3)
    st(line((1706, 2056), (1932, 2056), 10), 9, (128, 56, 70), (150, 70, 80), dry=0.2)
    st(line((1706, 2070), (1932, 2071), 10), 7, (96, 44, 60), dry=0.3)


def barn_mask(h, w):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    S = SCALE
    d.polygon([(x * S, y * S) for x, y in
               [(1000, 2050), (1200, 1986), (1565, 1992), (1565, 2078), (1000, 2078)]], fill=255)
    d.rectangle([1704 * S, 2040 * S, 1932 * S, 2076 * S], fill=255)
    return np.asarray(m) > 0


# ---------------------------------------------------------------- corn flicks

def corn_flicks(cv, ref):
    """Thin upward flicks of light yellow-green along the corn rows."""
    S = SCALE
    h, w, _ = ref.shape
    for _ in range(650):
        x = random.uniform(0, w)
        y = random.uniform(FIELD_TOP * S + 10, CORN_END * S)
        base = tuple(float(v) for v in ref[min(h - 1, int(y)), min(w - 1, int(x))])
        col = broken_color(mix(base, (226, 218, 150), random.uniform(0.15, 0.4)), 0.03, 0.05)
        ln = random.uniform(7, 17)
        a = -math.pi / 2 + random.gauss(0, 0.3)
        bend = random.gauss(0, 0.25)
        pts = [(x, y), (x + math.cos(a) * ln * 0.5, y + math.sin(a) * ln * 0.5),
               (x + math.cos(a + bend) * ln, y + math.sin(a + bend) * ln)]
        cv.flick(pts, random.choice([1, 2]), col, alpha=random.randint(120, 200))


# ---------------------------------------------------------------- impasto

def light_paint(cv):
    img = np.asarray(cv.img).astype(np.float32)
    hgt = ndimage.gaussian_filter(np.asarray(cv.height).astype(np.float32) / 255, 1.4)
    gy, gx = np.gradient(hgt)
    # light from the upper left
    shade = (-gx * 0.7 - gy * 0.7) * 1.6
    spec = np.clip(shade - 0.06, 0, 1) * 90
    img = img * (1 + np.clip(shade, -0.07, 0.07))[..., None] + spec[..., None]
    # canvas weave where the paint is thin
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    weave = (np.sin(xx * 1.9) * np.sin(yy * 1.9)) * 3 * (1 - np.clip(hgt * 2, 0, 1))
    img += weave[..., None] + np.random.normal(0, 2, img.shape[:2])[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="field.jpg")
    ap.add_argument("--out", default="field_painted.png")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    photo = Image.open(a.src).convert("RGB")
    W, H = int(photo.width * SCALE), int(photo.height * SCALE)
    ref = grade(np.asarray(photo.resize((W, H), Image.LANCZOS)).astype(np.float32), a.seed)

    m = 24                                  # paint past the borders, crop later
    refp = np.pad(ref, ((m, m), (m, m), (0, 0)), mode="reflect")
    lum = refp @ LUM
    rows = (np.arange(refp.shape[0]) - m) / SCALE
    trees = (lum < 115) & ((rows >= TREES_TOP) & (rows < FIELD_TOP + 4))[:, None]
    trees = ndimage.binary_dilation(trees, iterations=3)
    bm = np.zeros(refp.shape[:2], bool)
    bm[m:m + H, m:m + W] = barn_mask(H, W)

    cv = Canvas(refp.shape[1], refp.shape[0])
    # shift photo-space helpers by the margin
    cv.d = ImageDraw.Draw(cv.img, "RGBA")
    paint_layers_with_margin(cv, refp, bm, trees, a.seed, m)
    light_paint(cv).crop((m, m, m + W, m + H)).save(a.out)
    print("saved", a.out)


def paint_layers_with_margin(cv, refp, bm, trees, seed, m):
    """Region bands and hand strokes are in photo coordinates; offset them by
    the padding margin so they land in the right place on the padded canvas."""
    global region
    base_region = region

    def shifted_region(x, py, t):
        return base_region(x, py - m / SCALE, t)
    region = shifted_region
    paint_layers(cv, refp, bm, trees, seed)
    region = base_region

    # hand strokes and flicks: translate the draw calls by the margin
    orig = cv.stroke

    def stroke_shifted(pts, *args, **kw):
        orig([(x + m, y + m) for x, y in pts], *args, **kw)
    orig_flick = cv.flick

    def flick_shifted(pts, *args, **kw):
        orig_flick([(x + m, y + m) for x, y in pts], *args, **kw)
    cv.stroke, cv.flick = stroke_shifted, flick_shifted
    barn_strokes(cv)
    corn_flicks(cv, refp[m:-m, m:-m])
    cv.stroke, cv.flick = orig, orig_flick


if __name__ == "__main__":
    main()
