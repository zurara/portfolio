"""Procedural naive folk-art winter village, in the style of flat
Grandma Moses / Bruegel-like snow scenes.

Usage: python folk_winter.py [--seed N] [--out file.png]
"""
import argparse
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1440, 1040          # output size
S = 2                      # supersampling factor
HORIZON = 385              # y where mountains meet the valley mist

SKY_TOP = (70, 86, 76)
SKY_BOTTOM = (132, 148, 134)
MOUNTAIN = (84, 98, 88)
MOUNTAIN_DARK = (58, 70, 62)
SNOW = (236, 238, 233)
SNOW_SHADE = (214, 219, 216)
POND = (104, 160, 144)
POND_LIGHT = (150, 196, 182)
TRUNK = (40, 36, 33)
WALLS = [(118, 98, 82), (148, 58, 44), (92, 92, 88), (176, 166, 146),
         (66, 58, 54), (160, 82, 56), (132, 120, 104)]
COATS = [(44, 44, 50), (44, 44, 50), (150, 46, 40), (108, 104, 98),
         (98, 74, 56), (60, 56, 62), (186, 168, 140)]
SKIN = (214, 186, 160)


def P(v):
    """Scale 1x coordinates to the supersampled canvas."""
    return int(round(v * S))


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def jitter(c, amt=10):
    return tuple(max(0, min(255, v + random.randint(-amt, amt))) for v in c)


def perspective(y):
    """Object scale for a ground point at height y (0 at horizon, 1 at bottom)."""
    return max(0.08, (y - HORIZON + 25) / (H - HORIZON + 25))


class Canvas:
    def __init__(self):
        self.img = Image.new("RGB", (W * S, H * S), SNOW)
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def poly(self, pts, fill):
        self.d.polygon([(P(x), P(y)) for x, y in pts], fill=fill)

    def line(self, pts, fill, width):
        self.d.line([(P(x), P(y)) for x, y in pts], fill=fill,
                    width=max(1, P(width)), joint="curve")

    def ellipse(self, cx, cy, rx, ry, fill):
        self.d.ellipse([P(cx - rx), P(cy - ry), P(cx + rx), P(cy + ry)], fill=fill)

    def rect(self, x0, y0, x1, y1, fill):
        self.d.rectangle([P(x0), P(y0), P(x1), P(y1)], fill=fill)


# ---------------------------------------------------------------- sky + mountains

def draw_sky(cv):
    arr = np.zeros((H * S, W * S, 3), np.float32)
    t = np.linspace(0, 1, H * S)[:, None] ** 1.3
    top, bot = np.array(SKY_TOP, np.float32), np.array(SKY_BOTTOM, np.float32)
    arr[:] = (top + (bot - top) * t[..., None])
    cv.img.paste(Image.fromarray(arr.astype(np.uint8)), (0, 0))
    # soft horizontal brush streaks
    for _ in range(420):
        y = random.uniform(0, HORIZON)
        x = random.uniform(-100, W)
        ln = random.uniform(60, 300)
        shade = random.choice([(255, 255, 255, 10), (20, 30, 25, 12)])
        cv.line([(x, y), (x + ln, y + random.uniform(-3, 3))], shade, random.uniform(1, 4))


def ridge_profile(peaks, base, rough):
    xs = np.arange(W + 1, dtype=np.float32)
    ys = np.full_like(xs, base)
    for px, ph, pw in peaks:
        k = np.clip(1 - np.abs(xs - px) / pw, 0, 1) ** 1.35
        ys = np.minimum(ys, base - ph * k)
    # jagged crags
    noise = np.zeros_like(xs)
    amp = rough
    step = 64
    while step >= 2:
        pts = np.random.uniform(-amp, amp, W // step + 2)
        noise += np.interp(xs, np.arange(len(pts)) * step, pts)
        amp *= 0.55
        step //= 2
    smooth = ys.copy()
    return ys + noise * (base - ys) / max(1, base - ys.min()) * 1.2, smooth


def draw_mountains(cv):
    # back range, pale
    back, back_s = ridge_profile([(1180, 160, 260), (1330, 120, 200), (300, 90, 260),
                          (760, 110, 220)], HORIZON, 10)
    cv.poly([(0, HORIZON + 5)] + [(x, back[x]) for x in range(0, W + 1, 3)] +
            [(W, HORIZON + 5)], lerp(MOUNTAIN, SKY_BOTTOM, 0.55) + (255,))
    snow_streaks(cv, back, back_s, 900, 110)

    # main jagged massif
    front, front_s = ridge_profile([(640, 300, 150), (715, 330, 120), (560, 190, 130),
                           (820, 200, 130), (500, 120, 160), (930, 150, 170),
                           (1010, 110, 150)], HORIZON, 26)
    cv.poly([(0, HORIZON + 5)] + [(x, front[x]) for x in range(0, W + 1, 2)] +
            [(W, HORIZON + 5)], MOUNTAIN + (255,))
    # dark vertical gullies
    for _ in range(900):
        x = random.uniform(420, 1060)
        top = front[int(x)]
        if top > HORIZON - 8:
            continue
        y0 = top + random.uniform(0, (HORIZON - top) * 0.8)
        ln = random.uniform(6, 50)
        cv.line([(x, y0), (x + random.uniform(-6, 6), y0 + ln)],
                MOUNTAIN_DARK + (random.randint(60, 150),), random.uniform(0.6, 2))
    snow_streaks(cv, front, front_s, 2600, 230)

    # valley mist
    for i in range(60):
        y = HORIZON - 70 + i * 1.6
        a = int(255 * (i / 60) ** 1.8)
        cv.rect(0, y, W, y + 2, SNOW + (min(255, a),))


def snow_streaks(cv, ridge, smooth, n, alpha):
    """Snow lies in patches: strokes run straight down the fall line and
    cluster where a low-frequency noise field is high."""
    grad = np.gradient(np.convolve(smooth, np.ones(15) / 15, mode="same"))
    knots = np.arange(0, W + 60, 60)
    field = np.interp(np.arange(W + 1), knots, np.random.rand(len(knots)))
    for _ in range(n):
        x = random.uniform(0, W - 1)
        if random.random() > field[int(x)] ** 0.7:
            continue
        top = ridge[int(x)]
        depth = HORIZON - top
        if depth < 6:
            continue
        f = random.random() ** random.choice([0.5, 2.8])
        y0 = top + depth * f
        slope = float(np.clip(grad[int(x)], -2.5, 2.5))
        ln = random.uniform(5, 26)
        # down the fall line: away from the crest
        dx = -slope * ln * 0.35 + random.uniform(-1.5, 1.5)
        col = jitter(SNOW, 6) + (random.randint(alpha // 2, alpha),)
        cv.line([(x, y0), (x - dx, y0 + ln)], col, random.uniform(0.8, 2.6))
    # snow caps on the crests
    for x in range(0, W, 2):
        top = ridge[x]
        if HORIZON - top > 30:
            cv.line([(x, top + 1), (x - grad[x] * 3, top + random.uniform(3, 10))], SNOW + (170,), 1.6)


# ---------------------------------------------------------------- ground

def draw_ground(cv):
    cv.rect(0, HORIZON, W, H, SNOW + (255,))
    for _ in range(1400):
        y = random.uniform(HORIZON, H)
        x = random.uniform(-50, W)
        ln = random.uniform(20, 140) * perspective(y)
        col = random.choice([SNOW_SHADE + (70,), (255, 255, 255, 120), (200, 208, 210, 50)])
        cv.line([(x, y), (x + ln, y + random.uniform(-1, 1))], col, random.uniform(1, 3) * perspective(y))
    # foreground hill: a raised slope with a soft shadow along its crest
    crest = [(250, H), (520, 900), (900, 820), (1440, 700)]
    hill = crest + [(W, H)]
    cv.poly(hill, (246, 247, 244, 255))
    for i in range(12):
        off = i * 1.5
        cv.line([(x, y + off) for x, y in crest], (190, 198, 198, 22), 2)


def pond_mask_points(cx, cy, rx, ry, wob=0.08):
    pts = []
    phase = [random.uniform(0, 6.28) for _ in range(3)]
    for i in range(120):
        a = i / 120 * 2 * math.pi
        r = 1 + wob * (math.sin(3 * a + phase[0]) + 0.6 * math.sin(5 * a + phase[1])
                       + 0.3 * math.sin(9 * a + phase[2]))
        pts.append((cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)))
    return pts


def draw_pond(cv, cx, cy, rx, ry):
    pts = pond_mask_points(cx, cy, rx, ry)
    cv.poly(pts, POND + (255,))
    # snow rim
    cv.line(pts + [pts[0]], (240, 244, 240, 180), max(1.0, ry * 0.05))
    # skate scratches and sheen
    for _ in range(int(rx * ry / 40)):
        a = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.random()) * 0.9
        x, y = cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)
        ln = random.uniform(4, 30) * ry / 70
        col = random.choice([POND_LIGHT + (110,), (80, 130, 118, 90), (230, 240, 235, 60)])
        cv.line([(x, y), (x + ln, y + random.uniform(-0.6, 0.6))], col, random.uniform(0.6, 1.8))
    return pts


def inside(pts, x, y):
    c = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi:
            c = not c
        j = i
    return c


# ---------------------------------------------------------------- buildings

def draw_house(cv, x, y, s, wall=None, facing=None):
    """x, y = bottom-left of the gable end. s = pixel scale."""
    wall = wall or random.choice(WALLS)
    facing = facing if facing is not None else random.choice([-1, 1])
    gw = random.uniform(24, 34) * s           # gable width
    wh = random.uniform(18, 26) * s           # wall height
    rh = gw * random.uniform(0.55, 0.8)       # roof rise
    dl = random.uniform(26, 50) * s * facing  # side length (sign = which side)
    sk = -abs(dl) * 0.10                      # slight oblique rise of the side
    shade = lerp(wall, (20, 20, 20), 0.38)
    outline = (120, 124, 124, 200)

    gx0, gx1 = x, x + gw
    ax = x + gw / 2
    e = gx1 if facing > 0 else gx0            # the gable corner that meets the side
    side = [(e, y), (e + dl, y + sk), (e + dl, y + sk - wh), (e, y - wh)]
    cv.poly(side, wall + (255,))
    # dark strip under the eave
    cv.poly([(e, y - wh), (e + dl, y + sk - wh), (e + dl, y + sk - wh + 3 * s), (e, y - wh + 3 * s)],
            (30, 28, 26, 150))
    cv.poly([(gx0, y), (gx1, y), (gx1, y - wh), (ax, y - wh - rh), (gx0, y - wh)], shade + (255,))
    # snowy roof plane
    roof = [(ax, y - wh - rh), (ax + dl, y + sk - wh - rh),
            (e + dl + 2 * s * facing, y + sk - wh + 1.5 * s), (e + 2 * s * -facing * 0, y - wh + 1.5 * s)]
    cv.poly(roof, jitter(SNOW, 3) + (255,))
    # a little shading near the ridge, and outlines so white-on-white reads
    cv.poly([roof[0], roof[1], (roof[1][0], roof[1][1] + 2.5 * s), (roof[0][0], roof[0][1] + 2.5 * s)],
            SNOW_SHADE + (255,))
    cv.line(roof + [roof[0]], outline, max(0.5, 0.9 * s))
    cv.line([(gx0 - 2 * s, y - wh + 1.5 * s), (ax, y - wh - rh - 1 * s), (gx1 + 2 * s, y - wh + 1.5 * s)],
            SNOW + (255,), max(0.8, 3 * s))
    if s > 0.3:
        n = max(1, int(abs(dl) / (12 * s)))
        for i in range(n):
            if random.random() < 0.45:
                continue
            t = (i + 0.5) / n
            wx, wy = e + dl * t, y + sk * t - wh * 0.6
            cv.rect(wx - 2 * s, wy - 3 * s, wx + 2 * s, wy + 3 * s, (32, 30, 28, 220))
        if random.random() < 0.6:
            cv.rect(ax - 3 * s, y - 11 * s, ax + 3 * s, y, (30, 26, 24, 230))
    cv.line([(min(gx0, gx0 + dl) - 2 * s, y + 1 * s), (max(gx1, gx1 + dl) + 2 * s, y + sk / 2 + 1 * s)],
            SNOW + (255,), max(0.8, 2.4 * s))
    return (min(gx0, gx0 + dl), y - wh - rh, max(gx1, gx1 + dl), y + 2)


def draw_church(cv, x, y, s):
    draw_house(cv, x, y, s * 1.2, wall=(176, 170, 156), facing=1)
    tw, th = 12 * s, 58 * s
    cv.rect(x - tw, y - th, x, y, (164, 158, 146, 255))
    cv.poly([(x - tw - 1 * s, y - th), (x + 1 * s, y - th), (x - tw / 2, y - th - 22 * s)], SNOW + (255,))
    cv.rect(x - tw * 0.7, y - th + 8 * s, x - tw * 0.3, y - th + 16 * s, (40, 38, 36, 220))


def draw_haystack(cv, x, y, s):
    w, h = 14 * s, 13 * s
    cv.poly([(x - w, y), (x - w * 0.8, y - h * 0.7), (x, y - h), (x + w * 0.8, y - h * 0.7), (x + w, y)],
            (196, 180, 130, 255))
    cv.poly([(x - w * 0.6, y - h * 0.75), (x, y - h - 1 * s), (x + w * 0.6, y - h * 0.75)], SNOW + (230,))


def draw_fence(cv, pts, s0, s1, rails=2, post_gap=26):
    n = len(pts)
    total = [0]
    for i in range(1, n):
        total.append(total[-1] + math.dist(pts[i - 1], pts[i]))
    L = total[-1]

    def at(dist):
        for i in range(1, n):
            if total[i] >= dist:
                t = (dist - total[i - 1]) / (total[i] - total[i - 1] + 1e-9)
                return (pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t,
                        pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t)
        return pts[-1]

    d = 0.0
    tops = []
    while d <= L:
        t = d / L
        s = s0 + (s1 - s0) * t
        x, y = at(d)
        h = 30 * s
        cv.line([(x, y), (x + random.uniform(-0.5, 0.5), y - h)], TRUNK + (255,), max(0.6, 2.6 * s))
        tops.append((x, y, h, s))
        d += post_gap * s
    for r in range(rails):
        k = 0.35 + 0.5 * r
        cv.line([(x, y - h * k) for x, y, h, _ in tops], TRUNK + (235,), max(0.6, 1.6 * tops[0][3]))


# ---------------------------------------------------------------- figures

def draw_person(cv, x, y, h, coat=None, pose="stand", facing=1, hat=None, carry=None):
    coat = coat or random.choice(COATS)
    hat = hat or random.choice([(34, 32, 34), (34, 32, 34), (150, 46, 40), (120, 110, 100)])
    trousers = random.choice([(40, 38, 40), (60, 54, 50), (150, 46, 40)])
    lean = {"stand": 0, "walk": 0.08, "skate": 0.28, "bend": 0.45, "fallen": 1.3}[pose] * facing
    head_r = h * 0.09
    hip_y = y - h * 0.42
    lw = max(0.7, h * 0.07)
    # legs
    if pose == "skate":
        a, b = random.uniform(0.2, 0.45), random.uniform(-0.1, 0.1)
        feet = [(x - facing * h * a, y - h * 0.04), (x + facing * h * b, y)]
    elif pose == "walk":
        feet = [(x - h * 0.12, y), (x + h * 0.12, y)]
    elif pose == "fallen":
        feet = [(x - facing * h * 0.1, y - h * 0.25), (x + facing * h * 0.05, y - h * 0.3)]
        hip_y = y - h * 0.1
    else:
        feet = [(x - h * 0.06, y), (x + h * 0.06, y)]
    for fx, fy in feet:
        cv.line([(x, hip_y), (fx, fy)], trousers + (255,), lw)
        cv.line([(fx - h * 0.04 * facing, fy), (fx + h * 0.05 * facing, fy)], (28, 26, 26, 255), lw * 0.8)
    # torso direction
    tl = h * 0.4
    if pose == "fallen":
        sx, sy = x + facing * h * 0.4, hip_y - h * 0.08
    else:
        sx, sy = x + math.sin(lean) * tl, hip_y - math.cos(lean) * tl
    nx, ny = -(sy - hip_y), (sx - x)
    nl = math.hypot(nx, ny) or 1
    nx, ny = nx / nl, ny / nl
    wt, wb = h * 0.08, h * 0.13          # half widths at shoulder / hem
    hem_x, hem_y = x - (sx - x) * 0.25, hip_y - (sy - hip_y) * 0.25
    cv.poly([(sx + nx * wt, sy + ny * wt), (sx - nx * wt, sy - ny * wt),
             (hem_x - nx * wb, hem_y - ny * wb), (hem_x + nx * wb, hem_y + ny * wb)], coat + (255,))
    # arms
    arm = lw * 0.95
    if carry == "sticks":
        cv.line([(sx, sy + h * 0.02), (sx + facing * h * 0.12, sy + h * 0.18)], coat + (255,), arm)
        for k in range(4):
            ox = k * h * 0.04
            cv.line([(sx - facing * h * 0.05 + ox, sy + h * 0.1), (sx - facing * h * 0.12 + ox, sy - h * 0.45)],
                    (178, 138, 92, 255), lw * 0.25)
    else:
        swing = random.uniform(-0.5, 0.5) if pose in ("skate", "walk") else 0.1
        for sgn in (-1, 1):
            ax = sx + sgn * h * 0.2 * math.cos(swing) + facing * h * 0.05
            ay = sy + h * (0.22 if pose != "skate" else 0.1) * (1 + sgn * swing * 0.4)
            cv.line([(sx, sy + h * 0.02), (ax, ay)], coat + (255,), arm)
    # head and hat
    hx, hy = sx + (sx - x) * 0.25, sy - head_r * 1.05 + (0 if pose != "fallen" else head_r)
    if pose == "fallen":
        hx, hy = sx + facing * head_r * 1.2, sy - head_r * 0.2
    cv.ellipse(hx, hy, head_r, head_r, SKIN + (255,))
    cv.ellipse(hx, hy - head_r * 0.45, head_r * 1.05, head_r * 0.7, hat + (255,))
    if carry == "stick":
        cv.line([(sx + facing * h * 0.2, sy + h * 0.2), (sx + facing * h * 0.3, y)], TRUNK + (255,), lw * 0.6)


def draw_dog(cv, x, y, s, facing=1, col=None):
    col = col or random.choice([(36, 32, 30), (120, 96, 72), (80, 66, 54)])
    L, hgt = 26 * s, 10 * s
    lw = max(0.6, 2 * s)
    cv.line([(x - L / 2, y - hgt), (x + L / 2, y - hgt)], col + (255,), 5 * s)
    for dx, k in ((-L / 2, -1), (-L / 2 + 3 * s, 1), (L / 2 - 3 * s, -1), (L / 2, 1)):
        cv.line([(x + dx, y - hgt), (x + dx + k * 2 * s, y)], col + (255,), lw)
    hx = x + facing * (L / 2 + 3 * s)
    cv.line([(x + facing * L / 2, y - hgt), (hx, y - hgt - 5 * s), (hx + facing * 5 * s, y - hgt - 4 * s)],
            col + (255,), 3.5 * s)
    cv.poly([(hx - facing * 1 * s, y - hgt - 6 * s), (hx, y - hgt - 10 * s), (hx + facing * 1.5 * s, y - hgt - 6 * s)],
            col + (255,))
    cv.line([(x - facing * L / 2, y - hgt), (x - facing * (L / 2 + 8 * s), y - hgt - 4 * s)], col + (255,), lw)


def draw_sled(cv, x, y, s, rider=True):
    cv.line([(x - 14 * s, y), (x + 14 * s, y), (x + 18 * s, y - 4 * s)], (90, 70, 50, 255), 2 * s)
    cv.rect(x - 12 * s, y - 5 * s, x + 12 * s, y - 2 * s, (120, 90, 60, 255))
    if rider:
        cv.ellipse(x - 2 * s, y - 11 * s, 7 * s, 6 * s, (60, 48, 40, 255))
        cv.ellipse(x - 1 * s, y - 20 * s, 3.5 * s, 3.5 * s, SKIN + (255,))
        cv.ellipse(x - 1 * s, y - 22 * s, 3.8 * s, 2.5 * s, (40, 36, 36, 255))


def draw_bonfire(cv, x, y, s):
    for _ in range(40):
        a = random.uniform(-0.6, 0.6)
        ln = random.uniform(10, 26) * s
        col = random.choice([(242, 170, 50, 200), (226, 110, 40, 200), (250, 220, 120, 180)])
        cv.line([(x + random.uniform(-6, 6) * s, y), (x + math.sin(a) * ln, y - math.cos(a) * ln)], col, 2.5 * s)
    for k in (-1, 1):
        cv.line([(x - 9 * s * k, y + 1 * s), (x + 7 * s * k, y - 3 * s)], TRUNK + (255,), 2.2 * s)


# ---------------------------------------------------------------- trees

def branch(cv, x, y, ang, ln, w, depth):
    x2 = x + math.sin(ang) * ln
    y2 = y - math.cos(ang) * ln
    cv.line([(x, y), (x2, y2)], TRUNK + (255,), w)
    if depth == 0 or ln < 4:
        return
    # the leader keeps going, a side twig splits off
    branch(cv, x2, y2, ang * 0.85 + random.uniform(-0.2, 0.2), ln * random.uniform(0.7, 0.85), w * 0.7, depth - 1)
    if random.random() < 0.8:
        branch(cv, x2, y2, ang + random.choice([-1, 1]) * random.uniform(0.35, 0.7),
               ln * random.uniform(0.45, 0.65), w * 0.55, depth - 1)


def draw_tree(cv, x, base_y, top_y, width, spread=1.0, depth=6):
    """A tall bare tree: straight trunk, a few long forks in the upper part."""
    lean = random.uniform(-0.015, 0.015)
    height = base_y - top_y
    cv.poly([(x - width / 2, base_y), (x + width / 2, base_y),
             (x + width * 0.15 + lean * height, top_y), (x - width * 0.15 + lean * height, top_y)],
            TRUNK + (255,))
    t = 0.4 + random.uniform(0, 0.15)
    side = random.choice([-1, 1])
    while t < 0.95:
        y = base_y - height * t
        x0 = x + lean * height * t
        side = -side
        ang = side * random.uniform(0.3, 0.75) * spread
        ln = height * random.uniform(0.09, 0.17)
        w = width * (1 - t) * random.uniform(0.35, 0.5)
        branch(cv, x0, y, ang, ln, max(0.5, w), depth)
        t += random.uniform(0.04, 0.085)


def draw_small_tree(cv, x, y, h):
    cv.line([(x, y), (x, y - h)], TRUNK + (255,), max(0.6, h * 0.04))
    for _ in range(6):
        t = random.uniform(0.35, 0.95)
        ang = random.choice([-1, 1]) * random.uniform(0.3, 0.8)
        branch(cv, x, y - h * t, ang, h * 0.22, max(0.4, h * 0.02), 2)


def draw_birds(cv):
    for _ in range(4):
        x, y = random.uniform(500, 1400), random.uniform(220, 420)
        s = random.uniform(4, 9)
        cv.line([(x - s, y - s * 0.4), (x, y), (x + s, y - s * 0.4)], (30, 28, 28, 230), 1.4)


# ---------------------------------------------------------------- composition

def compose(seed):
    random.seed(seed)
    np.random.seed(seed)
    cv = Canvas()
    draw_sky(cv)
    draw_mountains(cv)
    draw_ground(cv)

    ponds = [
        (500, 720, 300, 72),   # big skating pond
        (470, 545, 95, 22),
        (945, 492, 105, 22),
    ]
    pond_pts = [pond_mask_points(*p, wob=0.0) for p in ponds]  # used for placement only

    def free(x0, y0, x1, y1, boxes):
        for px in (x0, x1, (x0 + x1) / 2):
            for py in (y0, y1):
                if any(inside(pts, px, py) for pts in pond_pts):
                    return False
        for b in boxes:
            if not (x1 < b[0] - 3 or x0 > b[2] + 3 or y1 < b[1] - 3 or y0 > b[3] + 3):
                return False
        return True

    # plan village objects, then draw from far to near
    items = []
    boxes = []
    for _ in range(2600):
        y = HORIZON + 30 * random.random() ** 0.6 + (230 * random.random() ** 1.6)
        x = random.uniform(-30, W)
        if y > 540 and 850 < x < 1440:     # keep the right mid-ground open
            continue
        s = perspective(y) * 1.05
        est = (x - 55 * s, y - 52 * s, x + 90 * s, y + 2)
        if free(*est, boxes):
            boxes.append(est)
            items.append(("house", x, y, s))
        if len(items) > 230:
            break
    # a few larger houses near the big pond and on the right
    for x, y in [(430, 840), (560, 845), (680, 830), (420, 890), (160, 870), (20, 840),
                 (840, 700), (950, 590), (1190, 530), (70, 690), (20, 640)]:
        items.append(("house", x, y, perspective(y) * 1.05))
    items.append(("church", 720, 540, perspective(540)))
    for _ in range(40):
        y = random.uniform(HORIZON + 20, 640)
        x = random.uniform(0, W)
        if not any(inside(pts, x, y) for pts in pond_pts):
            items.append(("hay", x, y, perspective(y)))
    for _ in range(35):
        y = random.uniform(HORIZON + 10, 560)
        items.append(("smalltree", random.uniform(0, W), y, perspective(y)))
    for _ in range(90):
        y = random.uniform(HORIZON + 15, 600)
        x = random.uniform(0, W)
        if not any(inside(pts, x, y) for pts in pond_pts):
            items.append(("walker", x, y, perspective(y)))

    # ponds sit on the ground, under everything that stands on it
    drawn = [draw_pond(cv, *p) for p in ponds]
    # skaters
    skaters = []
    for i, (cx, cy, rx, ry) in enumerate(ponds):
        n = [95, 14, 16][i]
        for _ in range(n):
            a = random.uniform(0, 2 * math.pi)
            r = math.sqrt(random.random()) * 0.85
            skaters.append((cy + ry * r * math.sin(a), cx + rx * r * math.cos(a)))
    # distant fences between fields
    for _ in range(10):
        y = random.uniform(HORIZON + 40, 600)
        x = random.uniform(0, W - 200)
        s = perspective(y)
        draw_fence(cv, [(x, y), (x + random.uniform(80, 200), y + random.uniform(-8, 8))], s, s, rails=1)

    ordered = sorted(items + [("skater", x, y, perspective(y)) for y, x in skaters], key=lambda it: it[2])
    for kind, x, y, s in ordered:
        if kind == "house":
            draw_house(cv, x, y, s)
        elif kind == "church":
            draw_church(cv, x, y, s)
        elif kind == "hay":
            draw_haystack(cv, x, y, s)
        elif kind == "smalltree":
            draw_small_tree(cv, x, y, 70 * s)
        elif kind == "walker":
            draw_person(cv, x, y, 34 * s, pose=random.choice(["stand", "walk"]), facing=random.choice([-1, 1]))
        elif kind == "skater":
            pose = random.choices(["skate", "stand", "fallen", "bend"], [70, 18, 5, 7])[0]
            draw_person(cv, x, y, 36 * s, pose=pose, facing=random.choice([-1, 1]))
        if y > 690 and kind == "house" and 780 < x < 900:
            pass

    # red barn and bonfire beside the pond
    draw_house(cv, 840, 732, 0.8, wall=(152, 62, 44), facing=1)
    draw_bonfire(cv, 818, 748, 0.9)
    for dx, f in ((-28, 1), (-14, 1), (26, -1), (40, -1)):
        draw_person(cv, 818 + dx, 752, 30, pose=random.choice(["stand", "bend"]), facing=f)

    # foreground hill: fence, path of people, dogs
    draw_fence(cv, [(470, 930), (800, 860), (1100, 810), (1415, 760)], 0.75, 1.1, rails=2, post_gap=28)
    crowd = [(560, 900, 58, "walk", 1, None), (610, 910, 52, "bend", -1, None), (640, 905, 50, "stand", 1, None),
             (705, 885, 56, "walk", 1, None), (755, 880, 64, "stand", -1, (150, 46, 40)),
             (800, 870, 50, "walk", 1, None), (830, 868, 64, "stand", 1, None),
             (870, 860, 62, "walk", -1, None), (945, 870, 66, "stand", 1, None),
             (1095, 855, 78, "stand", 1, None)]
    for x, y, h, pose, f, coat in crowd:
        draw_person(cv, x, y, h, coat=coat, pose=pose, facing=f)
    draw_sled(cv, 400, 972, 1.1)
    draw_person(cv, 470, 968, 66, pose="bend", facing=-1)
    draw_person(cv, 525, 968, 82, pose="walk", facing=-1)
    draw_person(cv, 290, 1010, 96, coat=(150, 138, 118), pose="stand", facing=1)
    draw_person(cv, 870, 975, 160, coat=(66, 60, 72), pose="stand", facing=1)
    draw_person(cv, 990, 1005, 190, coat=(40, 40, 44), pose="walk", facing=-1, carry="sticks")
    draw_person(cv, 1160, 985, 170, coat=(150, 46, 40), pose="bend", facing=-1, carry="stick")
    draw_dog(cv, 790, 990, 1.4, facing=1)
    draw_dog(cv, 950, 1015, 1.4, facing=-1)
    draw_dog(cv, 1080, 1030, 1.4, facing=1, col=(124, 98, 72))

    # reeds poking out of the snow
    for _ in range(90):
        x = random.uniform(0, 700)
        y = random.uniform(880, H)
        h = random.uniform(8, 24)
        for k in (-1, 0, 1):
            cv.line([(x, y), (x + k * h * 0.3, y - h)], (70, 64, 56, 180), 0.9)

    draw_birds(cv)
    # tall trees last
    for x, w, top in [(85, 26, -40), (215, 18, 60)]:
        draw_tree(cv, x, H + 10, top, w)
    for x, w, top in [(1245, 24, -40), (1290, 16, -40), (1335, 26, -40), (1415, 18, -40), (1180, 12, 150)]:
        draw_tree(cv, x, H - random.uniform(300, 360), top, w, spread=0.8)
    draw_tree(cv, 905, 845, -40, 15, spread=0.9)
    draw_tree(cv, 1010, 800, 40, 9, spread=0.9)
    draw_tree(cv, 330, 900, 160, 10)
    draw_tree(cv, 1100, 760, 80, 8, spread=0.8)
    return cv.img


def finish(img):
    img = img.resize((W, H), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    # canvas grain: fine noise plus faint horizontal weave
    grain = np.random.normal(0, 4.0, arr.shape[:2])[..., None]
    weave = (np.sin(np.arange(H) * 2.1)[:, None] * 1.2)[..., None]
    arr = np.clip(arr + grain + weave, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.SMOOTH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="folk_winter.png")
    a = ap.parse_args()
    finish(compose(a.seed)).save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
