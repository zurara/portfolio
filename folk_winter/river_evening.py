"""An invented scene, painted without any reference image.

"Evening by the river": a low sun behind thin cloud, a river bending towards
the viewer, a row of backlit poplars on the right bank, a village and two
haystacks in the distance, a red rowboat with a figure in a straw hat, and a
grassy bank with poppies in the left foreground.

Two stages, the way a painter works:
  1. build_reference(): a soft colour sketch of the scene (the "lay-in"),
     plus a label map saying what each pixel is (sky, water, poplar, ...).
  2. the stroke engine from paint_photo.py repaints it; each label gets its
     own brush handling, and the key objects (boat, figure, spire, haystacks,
     poplar rims, sun glitter, poppies) are stated with hand-placed strokes.

Usage: python river_evening.py [--out river_evening.png] [--seed N]
"""
import argparse
import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from paint_photo import Canvas, broken_color, light_paint, mix, structure_angle

W, H = 1600, 1100
HZ = 560                       # horizon
SUN = (1060, 455)
M = 24                         # painting margin, cropped at the end

SKY, CLOUD, FAR, VILLAGE, WATER, POPLAR, BANK, FORE = range(8)

BRUSHES = [30, 17, 9, 5]
THRESHOLD = [0, 16, 18, 22]


# ---------------------------------------------------------------- stage 1: the lay-in

def fbm(rng, h, w, sy, sx, octaves=4):
    """Anisotropic value noise: sy, sx are the feature sizes of the first octave."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for _ in range(octaves):
        g = rng.random((int(h / sy) + 3, int(w / sx) + 3)).astype(np.float32)
        z = ndimage.zoom(g, (sy, sx), order=3)[:h, :w]
        out += z * amp
        tot += amp
        amp *= 0.5
        sy, sx = max(2, sy / 2), max(2, sx / 2)
    return out / tot


def river_center(y):
    t = max(0.0, (y - (HZ + 15)) / (H - HZ - 15))
    return 1040 - 330 * t + 190 * math.sin(t * math.pi * 1.7), 10 + 330 * t ** 1.7


POPLARS = [(1180, 640, 380, 44), (1255, 655, 430, 50), (1340, 672, 490, 56),
           (1440, 690, 560, 62), (1548, 706, 610, 68)]


def poplar_halfwidth(f, width):
    """Flame silhouette: widest a third of the way up, pointed at the top."""
    if f < 0 or f > 1:
        return 0
    return width * math.sin(math.pi * min(1, f ** 0.75)) ** 0.9 * (1 - 0.25 * f)


def build_reference(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    label = np.full((H, W), SKY, np.uint8)
    C = lambda *c: np.array(c, np.float32)

    # --- sky: blue-violet above, warm yellow at the horizon, glow round the sun
    t = np.clip(yy / HZ, 0, 1)[..., None]
    sky = C(118, 132, 176) * (1 - t ** 0.8) + C(206, 188, 204) * t ** 0.8
    sky = sky * (1 - t ** 3) + C(250, 214, 158) * t ** 3
    d = np.hypot(xx - SUN[0], yy - SUN[1])[..., None]
    sky += C(60, 40, 5) * np.exp(-d / 200) + C(40, 40, 30) * np.exp(-(d / 45) ** 2)
    img[:] = sky

    # --- clouds: long horizontal bands; undersides lit warm, tops lilac
    n = fbm(rng, H, W, 34, 260)
    band = np.exp(-((yy - 330) / 170) ** 2)
    cm = np.clip((n * band - 0.30) / 0.12, 0, 1)
    below = np.clip((np.roll(n, -14, axis=0) - n) * 12 + 0.5, 0, 1)   # 1 = underside
    warmth = np.clip(np.exp(-d[..., 0] / 420) * 1.3, 0, 1)
    ccol = (C(168, 158, 186)[None, None] * (1 - below[..., None])
            + (C(236, 176, 160) * (1 - warmth[..., None]) + C(252, 206, 150) * warmth[..., None]) * below[..., None])
    sky_mask = yy < HZ
    k = (cm * sky_mask)[..., None]
    img = img * (1 - k) + ccol * k
    label[(cm > 0.35) & sky_mask] = CLOUD

    # --- distant hills and far bank
    hill = HZ - 26 - 14 * fbm(rng, 1, W, 1, 180, 3)[0] * 2
    hills = (yy > hill[None, :]) & (yy < HZ + 6)
    img[hills] = img[hills] * 0.35 + C(146, 140, 176) * 0.65
    label[hills] = FAR
    far = (yy >= HZ + 6) & (yy < 612)
    img[far] = C(168, 164, 112) * 0.7 + C(214, 190, 140) * 0.3
    label[far] = FAR
    # tree clumps on the far left
    clump = fbm(rng, H, W, 30, 60)
    trees = (xx < 470) & (yy > HZ - 40 + 60 * (clump - 0.5)) & (yy < 600) & (clump > 0.42)
    img[trees] = C(74, 88, 92)
    label[trees] = FAR

    # --- village with a church
    vr = random.Random(seed)
    for _ in range(16):
        x0 = vr.uniform(430, 640)
        w0, h0 = vr.uniform(14, 30), vr.uniform(10, 18)
        y0 = HZ + vr.uniform(-4, 6)
        wall = vr.choice([C(222, 190, 162), C(200, 170, 168), C(214, 200, 176)])
        m = (xx > x0) & (xx < x0 + w0) & (yy > y0 - h0) & (yy < y0)
        img[m] = wall
        r = (xx > x0 - 2) & (xx < x0 + w0 + 2) & (yy > y0 - h0 - 7) & (yy <= y0 - h0)
        img[r] = C(172, 112, 104)
        label[m | r] = VILLAGE
    tower = (xx > 538) & (xx < 560) & (yy > 480) & (yy < HZ + 4)
    img[tower] = C(208, 186, 170)
    label[tower] = VILLAGE

    for hx, hy, s in ((250, 604, 1.2), (336, 600, 0.95)):
        dome = (((xx - hx) / (20 * s)) ** 2 + ((yy - hy) / (26 * s)) ** 2 < 1) & (yy < hy)
        lit = dome & (xx < hx + 4 * s)
        img[dome] = C(160, 124, 130)
        img[lit] = C(232, 186, 132)
        label[dome] = VILLAGE
    # --- ground below the horizon: fields in bands, violet shadow patches
    ground = yy >= 612
    tg = np.clip((yy - 612) / (H - 612), 0, 1)[..., None]
    # contre-jour: warm, lit fields in the distance, cooler and darker near us
    g = C(178, 164, 98) * (1 - tg) + C(104, 124, 78) * tg
    stripe = np.sin((yy + 0.08 * xx) * 0.045 + 3 * fbm(rng, H, W, 40, 160)) * (1 - tg[..., 0])
    g = g + (C(40, 26, -6) * np.clip(stripe, 0, 1)[..., None] - C(20, 6, -10) * np.clip(-stripe, 0, 1)[..., None])
    g = g * (0.88 + 0.24 * fbm(rng, H, W, 26, 70)[..., None])
    shade = np.clip((fbm(rng, H, W, 70, 140) - 0.5) * 4, 0, 1)[..., None] * 0.45
    g = g * (1 - shade) + C(110, 108, 140) * shade
    img[ground] = g[ground]
    label[ground] = BANK

    cx = np.array([river_center(y)[0] for y in range(H)], np.float32)[:, None]
    hw = np.array([river_center(y)[1] for y in range(H)], np.float32)[:, None]
    # left foreground bank: rises towards the lower left, ends at the water
    bank_top = 740 + 160 * (xx / 700) ** 1.4 + 40 * (fbm(rng, H, W, 30, 80) - 0.5)
    fore = ground & (yy > bank_top) & (xx < cx - hw)
    fg = C(168, 158, 84) * (0.85 + 0.3 * fbm(rng, H, W, 40, 60)[..., None])
    img[fore] = fg[fore]
    label[fore] = FORE

    # --- river: reflects the sky, sun column, poplar reflections
    edge = 6 * (fbm(rng, H, W, 20, 20) - 0.5)
    water = (yy > HZ + 15) & (np.abs(xx - cx) < hw + edge)
    my = np.clip(HZ - (yy - HZ) * 0.7, 0, HZ - 1).astype(int)
    refl = img[my, xx.astype(int)]
    wcol = refl * 0.78 + C(64, 86, 128) * 0.22
    column = np.exp(-((xx - SUN[0]) / np.maximum(18, 18 + (yy - HZ) * 0.12)) ** 2)[..., None]
    wcol = wcol * (1 - column * 0.6) + C(255, 226, 160) * column * 0.6
    # the banks reflect as dark green along both edges; light ripples across
    dist_edge = hw + edge - np.abs(xx - cx)
    bank_refl = np.clip(1 - dist_edge / np.maximum(8, hw * 0.35), 0, 1)[..., None] * (yy > 640)[..., None]
    wcol = wcol * (1 - bank_refl * 0.6) + C(66, 84, 72) * bank_refl * 0.6
    ripple = np.clip((fbm(rng, H, W, 6, 90) - 0.55) * 5, 0, 1)[..., None] * 0.35
    wcol = wcol * (1 - ripple) + C(222, 214, 226) * ripple
    img[water] = wcol[water]
    label[water] = WATER
    margin = ndimage.binary_dilation(water, iterations=7) & ~water & ground
    img[margin] = img[margin] * 0.55 + C(70, 86, 70) * 0.45

    # --- poplars on the right bank, backlit, with their reflections
    for px, pb, ph, pw in POPLARS:
        f = (pb - yy) / ph
        hwp = np.vectorize(poplar_halfwidth)(np.clip(f[:, 0], -1, 2), pw)[:, None]
        wob = 7 * (fbm(rng, H, W, 18, 10) - 0.5)
        m = (np.abs(xx - px) < hwp + wob * (hwp > 2)) & (f > 0) & (f < 1)
        rim = m & (xx < px - hwp * 0.45)
        img[m] = C(46, 66, 62) * 0.8 + C(62, 70, 96) * 0.2
        img[rim] = C(150, 120, 70)
        label[m] = POPLAR
        # reflection: mirrored about the bank line, only on water
        fr = (yy - pb) / (ph * 0.8)
        hwr = np.vectorize(poplar_halfwidth)(np.clip(fr[:, 0], -1, 2), pw)[:, None]
        rm = (np.abs(xx - px) < hwr * 1.1) & (fr > 0) & water
        img[rm] = img[rm] * 0.45 + C(46, 62, 70) * 0.55
        # short cast shadow on the bank
        ell = ((xx - (px + 14)) / (pw * 1.3)) ** 2 + ((yy - (pb + 8)) / 13) ** 2
        k = (np.clip(1 - ell, 0, 1) ** 0.6 * ~water)[..., None] * 0.5
        img = img * (1 - k) + C(82, 78, 124) * k

    # overall soft focus: this is a lay-in, detail comes from the brush
    img = ndimage.gaussian_filter(img, sigma=(1.5, 1.5, 0))
    return np.clip(img, 0, 255), label, water


# ---------------------------------------------------------------- stage 2: brush handling

RULES = {
    #          mode     length curl spread  max_r
    SKY:     ("flat",   1.5,   0.3, 0.30,   None),
    CLOUD:   ("follow", 1.1,   1.0, 0.25,   None),
    FAR:     ("flat",   0.9,   0.1, 0.10,   9),
    VILLAGE: ("dab",    0.5,   0.0, 0.30,   5),
    WATER:   ("flat",   1.4,   0.0, 0.04,   None),
    POPLAR:  ("up",     1.1,   0.3, 0.18,   12),
    BANK:    ("flat",   1.8,   0.1, 0.18,   None),
    FORE:    ("grass",  0.9,   0.2, 0.50,   None),
}


def paint_auto(cv, ref, label, protect, seed):
    random.seed(seed)
    h, w, _ = ref.shape
    for layer, r in enumerate(BRUSHES):
        blurred = ndimage.gaussian_filter(ref, sigma=(r * 0.5, r * 0.5, 0))
        ang, strength = structure_angle(blurred, r * 1.2)
        cur = np.asarray(cv.img).astype(np.float32)
        err = np.sqrt(((cur - blurred) ** 2).sum(axis=2))
        if layer >= 1:
            err[protect] = 0
        grid = max(3, int(r))
        seeds = []
        for gy in range(0, h, grid):
            for gx in range(0, w, grid):
                cell = err[gy:gy + grid, gx:gx + grid]
                if cell.mean() > THRESHOLD[layer]:
                    iy, ix = np.unravel_index(np.argmax(cell), cell.shape)
                    seeds.append((gx + ix, gy + iy))
        random.shuffle(seeds)
        for x, y in seeds:
            mode, lmul, curl, spread, max_r = RULES[int(label[y, x])]
            rr = r * random.uniform(0.85, 1.1)
            if max_r:
                rr = min(rr, max_r)
            base = tuple(float(v) for v in blurred[y, x])
            if mode == "up":
                a0 = -math.pi / 2 + random.gauss(0, spread)
            elif mode == "grass":
                a0 = -math.pi / 2 + random.gauss(0, spread)
            elif mode == "dab":
                a0 = random.gauss(0, spread) + random.choice([0, math.pi / 2])
            elif mode == "follow" and strength[y, x] > 0.05:
                a0 = ang[y, x]
            else:
                a0 = random.gauss(0, spread) + (random.random() < 0.5) * math.pi
            pts, a, step = [(x, y)], a0, rr * 0.8
            for _ in range(max(2, int(lmul * random.uniform(3, 6)))):
                cx, cy = pts[-1]
                ix, iy = int(cx), int(cy)
                if not (0 <= ix < w and 0 <= iy < h):
                    break
                if len(pts) > 2 and (math.dist(blurred[iy, ix], base) > 36 or label[iy, ix] != label[y, x]):
                    break
                if curl > 0 and strength[iy, ix] > 0.05:
                    t = ang[iy, ix]
                    if math.cos(t - a) < 0:
                        t += math.pi
                    a += math.atan2(math.sin(t - a), math.cos(t - a)) * curl * 0.5
                a += random.gauss(0, 0.04)
                pts.append((cx + math.cos(a) * step, cy + math.sin(a) * step))
            nx = int(min(w - 1, max(0, x + math.sin(a0) * rr * 2)))
            ny = int(min(h - 1, max(0, y - math.cos(a0) * rr * 2)))
            cv.stroke(pts, rr, broken_color(base), broken_color(tuple(float(v) for v in blurred[ny, nx])),
                      alpha=250 if layer == 0 else 238, dry=0.25 if layer == 0 else 0.45)
        print(f"layer r={r:>2}: {len(seeds)} strokes")


# ---------------------------------------------------------------- stage 2b: stated strokes

def hand_strokes(cv, water, label):
    rnd = random.Random(7)

    def st(pts, r, col, col2=None, dry=0.3, alpha=245, jitter=True):
        c = broken_color(col, 0.01, 0.04, (1, 1.1)) if jitter else col
        cv.stroke([(x + M, y + M) for x, y in pts], r, c, col2, alpha=alpha, dry=dry, wobble=0.04)

    def line(p0, p1, n=8, sag=0.0):
        out = []
        for i in range(n + 1):
            t = i / n
            out.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t + sag * math.sin(math.pi * t)))
        return out

    # sun glitter: short bright horizontal touches down the sun column
    for _ in range(170):
        y = HZ + 18 + rnd.random() ** 1.6 * 260
        spread = 14 + (y - HZ) * 0.14
        x = SUN[0] + rnd.gauss(0, spread * 0.6)
        if not water[int(min(H - 1, y)), int(min(W - 1, max(0, x)))]:
            continue
        ln = rnd.uniform(6, 22) * (1 + (y - HZ) / 300)
        col = rnd.choice([(255, 238, 180), (255, 220, 150), (250, 200, 140), (255, 246, 210)])
        st(line((x - ln / 2, y), (x + ln / 2, y + rnd.uniform(-1, 1)), 3), rnd.uniform(1.5, 3.2), col, dry=0.5, alpha=230)

    # poplar rims: warm vertical strokes down the sunlit left edge
    for px, pb, ph, pw in POPLARS:
        for _ in range(9):
            f = rnd.uniform(0.12, 0.9)
            y = pb - f * ph
            x = px - poplar_halfwidth(f, pw) * rnd.uniform(0.55, 0.9)
            ln = ph * rnd.uniform(0.05, 0.1)
            col = rnd.choice([(232, 168, 92), (214, 150, 86), (240, 190, 110)])
            st(line((x, y + ln / 2), (x + rnd.uniform(-2, 2), y - ln / 2), 4), rnd.uniform(2.5, 4.5), col, dry=0.4, alpha=225)

    # church: shadow and light side of the tower, then the spire in one stroke
    st(line((545, HZ + 2), (545, 482), 6), 7, (222, 196, 170), (244, 216, 180), dry=0.1)
    st(line((556, HZ + 2), (556, 484), 6), 4, (150, 128, 154), dry=0.2)
    st(line((550, 484), (550, 462), 5), 6, (128, 96, 116), dry=0.1)
    st(line((550, 466), (550, 438), 5), 3, (128, 96, 116), dry=0.6)

    # haystacks on the far bank: cast shadow, then the dome in two curved strokes
    def arc(cx, cy, rx, ry, a0, a1, n=8):
        return [(cx + rx * math.cos(a0 + (a1 - a0) * i / n), cy - ry * math.sin(a0 + (a1 - a0) * i / n))
                for i in range(n + 1)]
    for hx, hy, s in ((250, 604, 1.2), (336, 600, 0.95)):
        st(line((hx - 6 * s, hy + 3 * s), (hx + 34 * s, hy + 8 * s), 5), 4.5 * s, (116, 108, 150), dry=0.4)
        st(arc(hx, hy, 12 * s, 18 * s, math.pi, math.pi * 0.45), 8 * s, (238, 190, 136), (250, 214, 160), dry=0.15)
        st(arc(hx + 2 * s, hy, 12 * s, 17 * s, 0, math.pi * 0.5), 7 * s, (160, 118, 134), dry=0.2)
        st(arc(hx - 2 * s, hy - 12 * s, 8 * s, 10 * s, math.pi * 0.9, math.pi * 0.3), 3 * s, (252, 222, 170), dry=0.4)

    # a little more light round the sun
    for _ in range(26):
        a = rnd.uniform(0, 2 * math.pi)
        rr = rnd.uniform(4, 34)
        x, y = SUN[0] + math.cos(a) * rr * 1.6, SUN[1] + math.sin(a) * rr * 0.7
        st(line((x - 12, y), (x + 12, y + rnd.uniform(-2, 2)), 4), rnd.uniform(3, 6),
           rnd.choice([(255, 246, 214), (255, 236, 180), (252, 226, 170)]), dry=0.5, alpha=200)

    # the boat: reflection first, then hull, gunwale, interior
    by = 842
    bx = int(river_center(by)[0]) - 20
    for i in range(5):
        y = by + 10 + i * 6
        ln = 120 - i * 16
        st(line((bx - ln / 2 + rnd.uniform(-8, 8), y), (bx + ln / 2 + rnd.uniform(-8, 8), y), 5),
           rnd.uniform(2, 3.2), mix((196, 92, 60), (70, 96, 130), 0.25 + i * 0.1), dry=0.6, alpha=210)
    st(line((bx - 78, by - 6), (bx + 76, by - 10), 10, sag=10), 9, (206, 86, 52), (228, 120, 70), dry=0.15)
    st(line((bx - 70, by + 4), (bx + 66, by + 1), 10, sag=6), 5, (140, 56, 50), dry=0.3)
    st(line((bx - 72, by - 12), (bx + 72, by - 16), 10, sag=8), 3, (60, 44, 58), dry=0.2)
    st(line((bx - 80, by - 15), (bx + 78, by - 19), 10, sag=9), 1.6, (246, 214, 170), dry=0.35)
    # oars
    st(line((bx + 6, by - 30), (bx - 70, by + 8), 6), 1.6, (110, 80, 60), dry=0.1)
    st(line((bx + 10, by - 30), (bx + 84, by + 4), 6), 1.6, (110, 80, 60), dry=0.1)
    # the figure: shirt (shadow + light), head, straw hat
    st(line((bx + 12, by - 18), (bx + 10, by - 48), 5), 8, (150, 160, 204), dry=0.1)
    st(line((bx + 6, by - 20), (bx + 5, by - 46), 5), 5, (236, 236, 246), dry=0.2)
    st(line((bx + 8, by - 52), (bx + 9, by - 58), 3), 4.5, (216, 164, 130), dry=0.1)
    st(line((bx - 4, by - 60), (bx + 22, by - 61), 5, sag=-2), 3.6, (242, 212, 120), dry=0.2)
    st(line((bx + 2, by - 64), (bx + 16, by - 64), 4), 3.2, (226, 190, 100), dry=0.2)

    # poppies and a few white flowers in the left foreground
    for _ in range(420):
        x = rnd.uniform(0, 900)
        y = rnd.uniform(760, H - 1)
        if label[int(y), int(x)] != FORE:
            continue
        s = 1 + (y - 830) / 270                    # bigger nearer the viewer
        col = rnd.choices([(222, 52, 40), (236, 84, 52), (246, 240, 224), (246, 212, 90)], [6, 3, 1, 1])[0]
        a = rnd.uniform(0, math.pi)
        ln = rnd.uniform(3, 7) * s
        st(line((x - math.cos(a) * ln / 2, y - math.sin(a) * ln / 2), (x + math.cos(a) * ln / 2, y + math.sin(a) * ln / 2), 3),
           rnd.uniform(2.2, 3.5) * s, col, dry=0.2, alpha=240)

    # reeds at the water's edge
    for _ in range(260):
        y = rnd.uniform(700, H)
        cxr, hwr = river_center(y)
        side = rnd.choice([-1, 1])
        x = cxr + side * (hwr + rnd.uniform(-4, 10))
        s = 0.6 + (y - 700) / 300
        ln = rnd.uniform(14, 34) * s
        a = -math.pi / 2 + rnd.gauss(0, 0.2) + side * 0.1
        col = rnd.choice([(84, 100, 62), (128, 130, 70), (70, 84, 66), (180, 160, 96)])
        pts = [(x + M, y + M), (x + M + math.cos(a) * ln, y + M + math.sin(a) * ln)]
        cv.flick(pts, rnd.choice([1, 2]), col, alpha=rnd.randint(160, 220))

    # grass blades in the foreground, drawn last over the flowers
    for _ in range(1100):
        x = rnd.uniform(0, 900)
        y = rnd.uniform(760, H - 1)
        if label[int(y), int(x)] != FORE:
            continue
        s = 1 + (y - 860) / 240
        ln = rnd.uniform(12, 30) * s
        a = -math.pi / 2 + rnd.gauss(0, 0.3)
        bend = rnd.gauss(0, 0.3)
        col = rnd.choice([(150, 160, 70), (196, 186, 96), (110, 128, 66), (214, 200, 120)])
        pts = [(x + M, y + M), (x + M + math.cos(a) * ln / 2, y + M + math.sin(a) * ln / 2),
               (x + M + math.cos(a + bend) * ln, y + M + math.sin(a + bend) * ln)]
        cv.flick(pts, rnd.choice([1, 2, 2]), col, alpha=rnd.randint(150, 220))


def protected_mask():
    """Where the hand strokes go: the auto layers only lay in colour there."""
    m = Image.new("L", (W + 2 * M, H + 2 * M), 0)
    d = ImageDraw.Draw(m)
    o = M
    bx = int(river_center(842)[0]) - 20
    d.rectangle([bx - 90 + o, 842 - 70 + o, bx + 90 + o, 842 + 40 + o], fill=255)   # boat
    d.rectangle([530 + o, 432 + o, 568 + o, HZ + 4 + o], fill=255)                 # church
    for hx, hy in ((250, 604), (336, 600)):
        d.rectangle([hx - 28 + o, hy - 34 + o, hx + 40 + o, hy + 12 + o], fill=255)
    return np.asarray(m) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="river_evening.png")
    ap.add_argument("--ref-out", default="river_evening_layin.png")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    ref, label, water = build_reference(a.seed)
    Image.fromarray(ref.astype(np.uint8)).save(a.ref_out)

    refp = np.pad(ref, ((M, M), (M, M), (0, 0)), mode="reflect")
    labp = np.pad(label, M, mode="edge")
    cv = Canvas(W + 2 * M, H + 2 * M)
    paint_auto(cv, refp, labp, protected_mask(), a.seed)
    hand_strokes(cv, water, label)
    light_paint(cv).crop((M, M, M + W, M + H)).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
