"""Build the portfolio's artwork from Dinesh's own footage.

Run:  python tools/build_content.py

Writes
    public/tools/*.png             twelve tool marks for the scene-2 card shell
    public/tools/figure.png        his silhouette for scene 2
    public/years/*.jpg             the six timeline cards + meta.json
    public/years/figure.png        his silhouette for scene 3
    public/projects/p01..p12.png   the project posters for scene 4
    public/projects/person.png     his figure for scene 4
    public/fin/plate.jpg           the closing wordmark + fog
    public/fin/man.png             his figure for the closing shot
    public/fin/frame2.jpg          the closing shot's second still

Everything is derived from the matted hero clip (public/media/hero.mp4, the
packed colour|alpha layout) and one grunge tile, so the whole site carries the
same grain, the same warm near-white and the same red as the film.
"""

import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(ROOT, "public", "media")
TOOLS = os.path.join(ROOT, "public", "tools")
YEARS = os.path.join(ROOT, "public", "years")
PROJ = os.path.join(ROOT, "public", "projects")
FIN = os.path.join(ROOT, "public", "fin")
TEX = os.path.join(ROOT, "public", "tex")

IMPACT = "C:/Windows/Fonts/impact.ttf"
UI = "C:/Windows/Fonts/seguisb.ttf"
UI_R = "C:/Windows/Fonts/segoeui.ttf"

WARM = (245, 233, 221)
PAPER = (216, 200, 186)
RED = (206, 44, 34)
RED_DEEP = (118, 22, 16)
MUTED = (150, 132, 122)
INK = (11, 10, 11)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def font(path, size):
    return ImageFont.truetype(path, max(6, int(round(size))))


def text_width(fnt, text, tracking=0.0):
    w = 0.0
    for ch in text:
        w += fnt.getlength(ch) + tracking
    return w - tracking if text else 0.0


def tracked(draw, xy, text, fnt, fill, tracking=0.0, anchor="lm"):
    """Letter-spaced text. PIL has no tracking, and this design needs it."""
    total = text_width(fnt, text, tracking)
    x, y = xy
    if anchor[0] == "m":
        x -= total * 0.5
    elif anchor[0] == "r":
        x -= total
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill, anchor="l" + anchor[1])
        x += fnt.getlength(ch) + tracking
    return total


def fit_size(fnt_path, text, max_w, max_size, min_size=8, tracking=0.0):
    """Largest size at which `text` (with tracking) fits `max_w`."""
    lo, hi = min_size, max_size
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if text_width(font(fnt_path, mid), text, tracking * mid) <= max_w:
            lo = mid
        else:
            hi = mid
    return lo


def rgba(w, h, col=(0, 0, 0, 0)):
    return Image.new("RGBA", (int(w), int(h)), col)


def grain(img, amount=10, seed=3):
    rng = np.random.default_rng(seed)
    a = np.asarray(img).astype(np.float32)
    n = rng.normal(0, amount, a.shape[:2])[..., None]
    rgb = a[..., :3] + n
    out = np.concatenate([np.clip(rgb, 0, 255), a[..., 3:]], -1)
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def vignette(img, power=1.0, floor=0.55):
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / w - 0.5) * 2
    ny = (yy / h - 0.5) * 2
    r = np.sqrt(nx * nx + ny * ny) / 1.414
    v = np.clip(1.0 - power * (r ** 2.1), floor, 1.0)[..., None]
    a = np.asarray(img).astype(np.float32)
    a[..., :3] *= v
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), img.mode)


def add_glow(img, centre, radius, colour, strength=1.0):
    """Additive radial light. Additive, not composited: a full-alpha layer
    would replace the surface it is meant to light."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / radius
    f = np.clip(1.0 - d, 0, 1) ** 2.2 * strength
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    a += f[..., None] * np.array(colour, np.float32)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# the matted clip
# --------------------------------------------------------------------------- #

class Clip:
    """Decode the packed colour|alpha clip and hand out RGBA stills."""

    def __init__(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h, w2 = fr.shape[:2]
            w = w2 // 2
            col = fr[:, :w].astype(np.float32)
            a = fr[:, w:, 0].astype(np.float32) / 255.0
            rgb = col / np.maximum(a[..., None], 1e-3)
            rgba_ = np.concatenate([np.clip(rgb, 0, 255), a[..., None] * 255.0], -1)
            frames.append(rgba_.astype(np.uint8))
        cap.release()
        if not frames:
            raise SystemExit("could not read %s" % path)
        self.frames = frames
        self.h, self.w = frames[0].shape[:2]

    def __len__(self):
        return len(self.frames)

    def at(self, i):
        """Frame as a PIL RGBA image, cropped to the subject with a margin."""
        i = int(np.clip(i, 0, len(self) - 1))
        img = Image.fromarray(self.frames[i], "RGBA")
        a = np.asarray(img)[..., 3]
        ys, xs = np.nonzero(a > 12)
        if len(ys) == 0:
            return img, (0, 0, img.width, img.height)
        pad = int(0.02 * img.width)
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
               min(img.width, xs.max() + pad), min(img.height, ys.max() + pad))
        return img.crop(box), box


def fit_rgba(img, w, h, anchor="bottom"):
    """Fit a cutout inside a canvas without distorting it."""
    s = min(w / img.width, h / img.height)
    nw, nh = max(1, int(img.width * s)), max(1, int(img.height * s))
    small = img.resize((nw, nh), Image.LANCZOS)
    out = rgba(w, h)
    x = (w - nw) // 2
    y = h - nh if anchor == "bottom" else (h - nh) // 2
    out.alpha_composite(small, (x, y))
    return out


def silhouette(img, tone=(30, 28, 29)):
    """Flat colour figure with the matte's own soft edge."""
    a = np.asarray(img).astype(np.float32)
    out = np.zeros_like(a)
    out[..., 0] = tone[0]
    out[..., 1] = tone[1]
    out[..., 2] = tone[2]
    out[..., 3] = a[..., 3]
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def still(img, w, h, zoom=1.0, cx=0.5, cy=0.5):
    """Cover-crop a cutout still onto an opaque canvas, subject centred."""
    a = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(a > 12)
    if len(xs) == 0:
        box = (0, 0, img.width, img.height)
    else:
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.04)
        box = (xs.min() - pad_x, ys.min() - pad_y,
               xs.max() + pad_x, ys.max() + pad_y)
        box = (max(0, box[0]), max(0, box[1]),
               min(img.width, box[2]), min(img.height, box[3]))
    crop = img.crop(box)
    s = max(w / crop.width, h / crop.height) * zoom
    nw, nh = int(crop.width * s), int(crop.height * s)
    crop = crop.resize((nw, nh), Image.LANCZOS)
    canvas = rgba(w, h, (0, 0, 0, 255))
    canvas.alpha_composite(crop, (int((w - nw) * cx), int((h - nh) * cy)))
    return canvas


def warm_subject(img, tint=(1.10, 0.82, 0.66), strength=0.55, lift=8.0):
    """Push a cutout into the site's warm light. The matted footage is lit
    cool — a grey studio key — and dropped into a red room untouched it reads
    as a sticker pasted on the plate. Blending toward a tinted luminance keeps
    his modelling but makes the light around him believable."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    rgb = a[..., :3]
    lum = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)
    toned = lum[..., None] * np.array(tint, np.float32) + lift
    a[..., :3] = np.clip(toned * strength + rgb * (1 - strength), 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def add_rim(img, colour=(255, 132, 74), strength=0.55, width=9):
    """A light wrap around the outline, from the matte's own soft edge."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    alpha = a[..., 3]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    grown = cv2.dilate(alpha, k)
    grown = cv2.GaussianBlur(grown, (0, 0), width * 0.5)
    edge = np.clip(grown - alpha, 0, 255)[..., None] / 255.0
    a[..., :3] = np.clip(
        a[..., :3] + edge * np.array(colour, np.float32) * strength, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def warm_grade(img, strength=0.55):
    """Pull a still into the film's warm near-monochrome red."""
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    lum = (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114)[..., None]
    tint = np.array([1.06, 0.72, 0.62], np.float32)
    graded = np.clip(lum * tint, 0, 1) ** 1.08
    mix = graded * strength + a * (1 - strength)
    return Image.fromarray((np.clip(mix, 0, 1) * 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- #
# 1  the twelve tool marks
# --------------------------------------------------------------------------- #

TOOL_SET = [
    ("meta", "META ADS", "Acquisition · Retargeting"),
    ("googleads", "GOOGLE ADS", "Search · YouTube · PMax"),
    ("gmb", "BUSINESS PROFILE", "Local visibility"),
    ("seo", "LOCAL SEO", "Citations · Rankings"),
    ("botpress", "BOTPRESS", "WhatsApp · Instagram"),
    ("vapi", "VAPI", "Voice agents"),
    ("davinci", "DAVINCI RESOLVE", "Edit · Grade · Deliver"),
    ("photoshop", "PHOTOSHOP", "Design · Retouch"),
    ("nextjs", "NEXT.JS", "App development"),
    ("claude", "CLAUDE", "AI-assisted builds"),
    ("chatgpt", "CHATGPT", "Copy · Automation"),
    ("analytics", "ANALYTICS", "GA4 · Reporting"),
]


def build_tools():
    os.makedirs(TOOLS, exist_ok=True)
    for name, title, _sub in TOOL_SET:
        W, H = 640, 400
        img = rgba(W, H)
        d = ImageDraw.Draw(img)
        accent = RED
        # the mark sits on the card's graphite slab, so it is drawn light and
        # leaves everything around it transparent
        # A card face is only ~90-160px on screen, so it carries the mark and
        # nothing else: the category lives in the section's screen-reader list,
        # not in six-pixel type.
        size = fit_size(IMPACT, title, W * 0.88, int(H * 0.52), 40,
                        tracking=0.012)
        f = font(IMPACT, size)
        tracked(d, (W * 0.5, H * 0.47), title, f, WARM + (255,),
                0.012 * size, anchor="mm")
        rule_w = W * 0.13
        y = H * 0.66
        d.rectangle([W * 0.5 - rule_w / 2, y, W * 0.5 + rule_w / 2, y + 3],
                    fill=accent + (235,))
        grain(img, 3, seed=hash(name) & 0xffff).save(
            os.path.join(TOOLS, name + ".png"))

    # the twelve card slots are named in cards.js; anything left over from the
    # reference set is stale and must not survive
    keep = {n + ".png" for n, _t, _s in TOOL_SET} | {"figure.png", "logos.json"}
    for f in os.listdir(TOOLS):
        if f not in keep:
            os.remove(os.path.join(TOOLS, f))
    print("  tools: %d marks" % len(TOOL_SET))


# --------------------------------------------------------------------------- #
# 2  posters for scene four
# --------------------------------------------------------------------------- #

PROJECTS = [
    ("p01_soulsync", "SoulSync", "WhatsApp AI · Product",
     "A dating app that runs inside WhatsApp, built with AI and shipped on Netlify."),
    ("p02_podcast", "Health Care Talks", "Podcast · Studio",
     "Studio direction, three-camera set-ups and every episode cut in Resolve."),
    ("p03_jewellery", "Fefa Jewellery", "3D showcase · Reels",
     "A 3D storefront and a reel engine for a jewellery house."),
    ("p04_rex", "Rex Hospitals", "Patient acquisition · Meta",
     "Performance campaigns and medical video for a multi-speciality hospital."),
    ("p05_heartlink", "Heartlink Hospitals", "Local SEO · GMB",
     "Citations, local listings and search visibility that fills the OPD."),
    ("p06_caayara", "Caayara Mobiles", "WhatsApp bot · Instagram",
     "Botpress flows on WhatsApp and Instagram, wired to the sales team."),
    ("p07_osdev", "OS Developers", "Real estate · Web",
     "Property campaigns, landing pages and walkthrough films."),
    ("p08_buildingblocks", "Building Blocks", "Instant Forms · Meta",
     "Meta instant forms that turn scrolls into qualified site visits."),
    ("p09_growthora", "Growthora", "Brand · Web · Video",
     "The studio site: positioning, design, build and the launch film."),
    ("p10_reels", "Reels & Films", "DaVinci · Brand video",
     "Short-form that people actually finish, cut and graded by hand."),
    ("p11_medical", "Medical Video", "Script · Shoot · Edit",
     "Doctor-led explainers shot on a ZV-E10 and cut for trust."),
    ("p12_automation", "AI Automations", "Botpress · Vapi · Claude",
     "Voice and chat agents that answer, book and hand over."),
]

BOX = {
    "p01_soulsync": (384, 273), "p02_podcast": (428, 269),
    "p03_jewellery": (471, 288), "p04_rex": (460, 377),
    "p05_heartlink": (232, 198), "p06_caayara": (173, 203),
    "p07_osdev": (211, 204), "p08_buildingblocks": (195, 206),
    "p09_growthora": (231, 208), "p10_reels": (355, 242),
    "p11_medical": (345, 243), "p12_automation": (325, 234),
}

OLD_PROJECT_FILES = {
    "p01_timeless", "p02_game", "p03_build", "p04_driven", "p05_ideas",
    "p06_sound", "p07_food", "p08_travel", "p09_space", "p10_play",
    "p11_cleaner", "p12_steps",
}


def motif(d, W, H, i, colour):
    """A watchable geometric mark, one family per card."""
    rng = random.Random(i * 977)
    cx, cy = W * 0.70, H * 0.40
    kind = i % 6
    if kind == 0:                                  # concentric arcs
        for k in range(6):
            r = min(W, H) * (0.16 + k * 0.085)
            d.arc([cx - r, cy - r, cx + r, cy + r], 200, 340,
                  fill=colour, width=max(1, int(W * 0.006)))
    elif kind == 1:                                # dot grid
        step = max(6, int(W * 0.035))
        for yy in range(int(H * 0.14), int(H * 0.72), step):
            for xx in range(int(W * 0.42), int(W * 0.94), step):
                d.ellipse([xx, yy, xx + 2, yy + 2], fill=colour)
    elif kind == 2:                                # waveform
        for k in range(26):
            x = W * 0.44 + k * W * 0.019
            h2 = (0.5 + 0.5 * math.sin(k * 0.8 + i)) * H * 0.22
            d.rectangle([x, cy - h2 / 2, x + W * 0.008, cy + h2 / 2],
                        fill=colour)
    elif kind == 3:                                # rising bars
        for k in range(7):
            h2 = H * (0.06 + 0.055 * k)
            x = W * 0.52 + k * W * 0.058
            d.rectangle([x, H * 0.62 - h2, x + W * 0.034, H * 0.62],
                        fill=colour)
    elif kind == 4:                                # orbit
        r = min(W, H) * 0.26
        d.ellipse([cx - r, cy - r * 0.62, cx + r, cy + r * 0.62],
                  outline=colour, width=max(1, int(W * 0.005)))
        d.ellipse([cx - r * 0.10, cy - r * 0.72, cx + r * 0.10, cy - r * 0.48],
                  fill=colour)
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=colour)
    else:                                          # window frames
        for k in range(4):
            x = W * 0.46 + k * W * 0.11
            d.rectangle([x, H * 0.16, x + W * 0.075, H * 0.62],
                        outline=colour, width=max(1, int(W * 0.005)))
    if rng.random() < 1:
        pass


def poster(W, H, i, title, kicker):
    img = Image.new("RGB", (W, H), INK)
    # deep red bloom right, cool key top-left
    img = add_glow(img, (W * 0.68, H * 0.36), max(W, H) * 0.80,
                   (150, 34, 26), 0.62)
    img = add_glow(img, (W * 0.10, H * 0.06), max(W, H) * 0.65,
                   (26, 27, 34), 0.85)
    d = ImageDraw.Draw(img, "RGBA")
    motif(d, W, H, i, (226, 120, 102, 60))

    # frame line
    m = max(3, int(W * 0.028))
    d.rectangle([m, m, W - m, H - m], outline=(236, 214, 198, 40),
                width=max(1, int(W * 0.004)))

    # header: index and category on one tracked line, clear of the mark and
    # of the title block at the foot of the card
    pad = W * 0.035
    maxw = W - 2 * m - 2 * pad
    head = "%02d   %s" % (i + 1, kicker.upper())
    hs = fit_size(UI, head, maxw, max(10, int(H * 0.062)), 9, 0.26)
    tracked(d, (m + pad, m + H * 0.075), head, font(UI, hs),
            (226, 108, 88, 235), hs * 0.26)

    # title, wrapped to at most three lines
    words = title.upper().split()
    lines, cur = [], ""
    size = int(H * 0.155)
    while size > 10:
        f = font(IMPACT, size)
        lines, cur = [], ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if text_width(f, t) <= maxw:
                cur = t
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        if len(lines) <= 3 and all(text_width(f, t) <= maxw for t in lines):
            break
        size = int(size * 0.92)
    f = font(IMPACT, size)
    lead = size * 1.06
    base = H - m - H * 0.13
    for k, line in enumerate(reversed(lines)):
        tracked(d, (m + pad, base - k * lead), line, f,
                WARM + (255,), 0.004 * size, anchor="ls")

    img = grain(img.convert("RGBA"), 6, seed=i * 31 + 5).convert("RGB")
    return vignette(img.convert("RGBA"), 0.45, 0.62).convert("RGB")


def build_projects(clip):
    os.makedirs(PROJ, exist_ok=True)
    for i, (pid, name, kicker, _desc) in enumerate(PROJECTS):
        bw, bh = BOX[pid]
        W, H = int(bw * 2), int(bh * 2)
        poster(W, H, i, name, kicker).save(
            os.path.join(PROJ, pid + ".png"), optimize=True)

    for f in os.listdir(PROJ):
        stem = os.path.splitext(f)[0]
        if stem in OLD_PROJECT_FILES:
            os.remove(os.path.join(PROJ, f))

    # his figure for the room: the same box the reference used
    fig, _box = clip.at(len(clip) * 0.55)
    figure = fit_rgba(fig, 160, 335)
    figure = warm_subject(figure, (1.16, 0.80, 0.62), 0.72, 10.0)
    add_rim(figure, (255, 150, 92), 0.22, 5).save(
        os.path.join(PROJ, "person.png"))
    print("  projects: %d posters + person" % len(PROJECTS))


# --------------------------------------------------------------------------- #
# 3  the timeline cards
# --------------------------------------------------------------------------- #

CHAPTERS = [
    (2021, "Foundations", ["Nursing college", "First edits", "A borrowed camera"]),
    (2022, "The craft", ["Shooting video", "Cutting reels", "Learning design"]),
    (2023, "Practice", ["Freelance edits", "Photoshop nights", "DaVinci Resolve"]),
    (2024, "Camera & cut", ["Sony ZV-E10", "Brand videos", "First campaigns"]),
    (2025, "Performance", ["Meta ads", "Instant Forms", "Local SEO"]),
    (2026, "Building now", ["Six brands", "AI-built apps", "SoulSync live"]),
]


def build_years(clip):
    os.makedirs(YEARS, exist_ok=True)
    meta = {}
    n = len(clip)
    for k, (year, _key, _lines) in enumerate(CHAPTERS):
        W, H = 528, 406 if k % 3 == 0 else 380
        frame, _box = clip.at(n * (0.12 + k * 0.15))
        img = still(frame, W, H, zoom=1.55 + 0.06 * (k % 3),
                    cx=0.5, cy=0.0).convert("RGB")
        img = warm_grade(img, 0.62)
        # darken the bottom so the type sits on something
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        for y in range(H):
            a = int(210 * max(0.0, (y / H - 0.28)) ** 1.4)
            sd.line([(0, y), (W, y)], fill=(8, 6, 6, a))
        # the card prints the year and the chapter itself, so the artwork is
        # only the photograph: type inside the still would say it twice
        img = Image.alpha_composite(img.convert("RGBA"), shade)
        out = vignette(grain(img, 6, seed=year).convert("RGB").convert("RGBA"),
                       0.4, 0.7).convert("RGB")
        out.save(os.path.join(YEARS, "%d.jpg" % year), quality=88)
        meta[str(year)] = {"w": W, "h": H}

    fig, _box = clip.at(n * 0.55)
    fit_rgba(fig, 123 * 2, 317 * 2).save(os.path.join(YEARS, "figure.png"))
    meta["figure"] = {"w": 123 * 2, "h": 317 * 2}
    with open(os.path.join(YEARS, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("  years: %d cards" % len(CHAPTERS))


# --------------------------------------------------------------------------- #
# 4  the closing shot
# --------------------------------------------------------------------------- #

def grunge_mask(W, H, seed=2):
    g = Image.open(os.path.join(TEX, "grunge.png")).convert("L")
    g = g.resize((W, H), Image.LANCZOS)
    return np.asarray(g).astype(np.float32) / 255.0


def fog_layer(W, H, seed=9):
    rng = np.random.default_rng(seed)
    n = rng.normal(0.5, 0.28, (H // 4, W // 4)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 26.0)
    n = (n - n.min()) / max(n.max() - n.min(), 1e-6)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(n, 0, 1)


def build_finale(clip):
    os.makedirs(FIN, exist_ok=True)
    W, H = 1600, 900
    # ---- the plate: furnace behind the wordmark ---------------------------
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    base = (np.array([9, 4, 4], np.float32) * (1 - yy)
            + np.array([26, 7, 5], np.float32) * yy)
    plate = np.repeat(base[:, None, :], W, axis=1)

    plate_img = add_glow(Image.fromarray(plate.astype(np.uint8), "RGB"),
                         (W * 0.50, H * 0.44), W * 0.68, (168, 30, 20), 1.05)

    # wordmark: bright enough that the finale shader's luminance gate lifts it
    # out of the fog exactly as the reference type does
    word = rgba(W, H)
    d = ImageDraw.Draw(word)
    size = fit_size(IMPACT, "DINESH", W * 0.68, int(H * 0.42), 40, 0.006)
    f = font(IMPACT, size)
    tracked(d, (W * 0.5, H * 0.47), "DINESH", f, (255, 224, 204, 255),
            0.006 * size, anchor="mm")
    # distress the letters with the same surface the hero wordmark uses
    wa = np.asarray(word).astype(np.float32)
    g = grunge_mask(W, H)
    g = (g - g.mean()) * 1.5 + 0.92
    wa[..., :3] *= np.clip(g, 0.90, 1.08)[..., None]
    word = Image.fromarray(np.clip(wa, 0, 255).astype(np.uint8), "RGBA")

    # a soft bloom of the letters themselves, so they glow through the smoke
    halo = word.filter(ImageFilter.GaussianBlur(size * 0.06))
    halo_a = np.asarray(halo).astype(np.float32)
    halo_a[..., :3] *= np.array([0.55, 0.14, 0.09], np.float32)
    halo = Image.fromarray(np.clip(halo_a, 0, 255).astype(np.uint8), "RGBA")

    plate_img = plate_img.convert("RGBA")
    plate_img = Image.alpha_composite(plate_img, halo)
    plate_img = Image.alpha_composite(plate_img, word)

    # fog: two octaves, screen-blended low and slow
    fog = fog_layer(W, H)
    fog2 = fog_layer(W, H, seed=21)
    mass = np.clip(fog * 0.65 + fog2 * 0.35, 0, 1)
    mass = np.clip((mass - 0.35) * 1.9, 0, 1) ** 1.5
    pa = np.asarray(plate_img.convert("RGB")).astype(np.float32)
    pa += mass[..., None] * np.array([74, 13, 9], np.float32) * 0.55
    plate_img = Image.fromarray(np.clip(pa, 0, 255).astype(np.uint8), "RGB")
    plate_img = vignette(grain(plate_img.convert("RGBA"), 8, seed=4)
                         .convert("RGB").convert("RGBA"), 0.55, 0.35)
    plate_img.convert("RGB").save(os.path.join(FIN, "plate.jpg"), quality=88,
                                  subsampling=1)

    # ---- the man ---------------------------------------------------------
    frame, _box = clip.at(len(clip) * 0.46)
    man = fit_rgba(frame, 360, 560)
    man = warm_subject(man, (1.18, 0.64, 0.48), 0.70, 4.0)
    man = add_rim(man, (255, 110, 62), 0.28, 5)
    man.save(os.path.join(FIN, "man.png"))
    with open(os.path.join(FIN, "fin.json"), "w", encoding="utf-8") as f:
        json.dump({"man": [628, 302, 360, 560], "size": [W, H]}, f)

    # ---- the second still: a red stage for the close ----------------------
    W2, H2 = 1600, 900
    floor_y = H2 * 0.80
    stage = Image.new("RGB", (W2, H2), (22, 7, 6))
    d = ImageDraw.Draw(stage)
    # a cream wedge slicing the frame, the red wall behind him, a dark floor
    d.polygon([(0, 0), (W2 * 0.52, 0), (W2 * 0.15, floor_y), (0, floor_y)],
              fill=(236, 226, 212))
    d.polygon([(W2 * 0.52, 0), (W2, 0), (W2, floor_y), (W2 * 0.15, floor_y)],
              fill=(178, 24, 19))
    fd = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    fdr = ImageDraw.Draw(fd)
    for y in range(int(floor_y), H2):
        t = (y - floor_y) / max(1.0, H2 - floor_y)
        fdr.line([(0, y), (W2, y)],
                 fill=(88, 12, 10, int(255 * (0.55 + 0.45 * (1 - t)))))
    stage = Image.alpha_composite(stage.convert("RGBA"), fd).convert("RGB")
    stage = add_glow(stage, (W2 * 0.80, H2 * 0.30), W2 * 0.66, (120, 26, 16), 0.85)
    stage = add_glow(stage, (W2 * 0.22, H2 * 0.14), W2 * 0.40, (96, 84, 72), 0.55)
    stage = add_glow(stage, (W2 * 0.66, H2 * 0.92), W2 * 0.34, (120, 30, 20), 0.45)

    # his shadow first, so he stands ON the floor rather than over it
    sh = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([W2 * 0.575, H2 * 0.862, W2 * 0.725, H2 * 0.912],
                               fill=(40, 3, 2, 215))
    stage = Image.alpha_composite(stage.convert("RGBA"),
                                  sh.filter(ImageFilter.GaussianBlur(14)))

    body, _box = clip.at(len(clip) * 0.30)
    bw = int(W2 * 0.26)
    bh = int(H2 * 0.62)
    b = np.asarray(fit_rgba(body, bw, bh)).astype(np.float32)
    # grade him into the red room: the wall's bounce is in every highlight
    rgb = b[..., :3]
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    tone = np.stack([lum * 1.16 + 30, lum * 0.74 + 8, lum * 0.68 + 6], -1)
    b[..., :3] = np.clip(tone * 0.62 + rgb * 0.38, 0, 255)
    body_img = Image.fromarray(b.astype(np.uint8), "RGBA")
    body_img = add_rim(body_img, (255, 128, 78), 0.35, 7)

    feet_y = int(floor_y + H2 * 0.088)
    twin = int(W2 * 0.655 - bw / 2)

    # a wet-floor reflection: the plate is a lacquer surface, so he casts one
    refl = body_img.transpose(Image.FLIP_TOP_BOTTOM)
    ra = np.asarray(refl).astype(np.float32)
    fade = np.clip(1.0 - np.linspace(0, 1, ra.shape[0]), 0, 1) ** 1.6
    ra[..., 3] *= fade[:, None] * 0.30
    ra[..., :3] *= 0.55
    refl = Image.fromarray(np.clip(ra, 0, 255).astype(np.uint8), "RGBA")
    refl = refl.filter(ImageFilter.GaussianBlur(3))

    lay = stage.convert("RGBA")
    lay.alpha_composite(refl, (twin, feet_y))
    lay.alpha_composite(body_img, (twin, feet_y - bh))
    stage = lay
    frame2 = vignette(grain(stage, 7, seed=12), 0.5, 0.45).convert("RGB")
    frame2.save(os.path.join(FIN, "frame2.jpg"), quality=88, subsampling=1)
    print("  finale: plate + man + frame2")


# --------------------------------------------------------------------------- #
# 5  the scene-2 / scene-3 silhouettes
# --------------------------------------------------------------------------- #

def build_figures(clip):
    fig, _box = clip.at(len(clip) * 0.60)
    fit_rgba(fig, 300, 720).save(os.path.join(TOOLS, "figure.png"))
    print("  figures: scene 2 silhouette")


def main():
    clip = Clip(os.path.join(MEDIA, "hero.mp4"))
    print("clip: %d frames %dx%d" % (len(clip), clip.w, clip.h))
    build_tools()
    build_figures(clip)
    build_years(clip)
    build_projects(clip)
    build_finale(clip)
    print("done")


if __name__ == "__main__":
    main()
