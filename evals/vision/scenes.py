"""
scenes.py — Synthetic wound photos with exact ground truth.

Each scene is a skin-toned canvas with an irregular wound (granulation plus
slough patches) and, optionally, a US quarter drawn at the correct size for
the simulated camera distance. Because the wound is rasterized by us, its
true area in cm² is known exactly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from vision import COIN_REAL_DIAM_CM

# Skin tones (BGR), light → dark
SKIN_TONES: dict[str, tuple[int, int, int]] = {
    "light":  (150, 170, 215),
    "fair":   (110, 130, 180),
    "medium": (85, 115, 165),
    "tan":    (70, 100, 145),
    "dark":   (45, 65, 100),
}

GRANULATION_BGR = (40, 50, 200)
SLOUGH_BGR = (40, 200, 225)

# The pipeline's no-coin fallback assumes 0.026 cm/px ≈ 38.5 px/cm.
# Simulated distances span half to double that resolution.
PX_PER_CM_RANGE = (19.0, 77.0)
IMG_H, IMG_W = 720, 960


@dataclass
class Scene:
    image: np.ndarray            # BGR photo with coin
    image_no_coin: np.ndarray    # same photo, coin removed
    true_area_cm2: float
    px_per_cm: float
    coin_center: tuple[int, int]
    coin_radius_px: float
    skin_tone: str


def _wound_polygon(rng: random.Random, center: tuple[int, int], radius_px: float) -> np.ndarray:
    """Irregular blob: an ellipse with smooth random radial perturbation."""
    aspect = rng.uniform(0.55, 1.0)
    rotation = rng.uniform(0, math.pi)
    harmonics = [(k, rng.uniform(0, 0.12), rng.uniform(0, 2 * math.pi)) for k in (2, 3, 5)]
    points = []
    for i in range(72):
        t = 2 * math.pi * i / 72
        r = radius_px * (1 + sum(a * math.sin(k * t + ph) for k, a, ph in harmonics))
        x, y = r * math.cos(t), r * aspect * math.sin(t)
        xr = x * math.cos(rotation) - y * math.sin(rotation)
        yr = x * math.sin(rotation) + y * math.cos(rotation)
        points.append((center[0] + xr, center[1] + yr))
    return np.round(points).astype(np.int32)


def _draw_coin(img: np.ndarray, center: tuple[int, int], radius: float) -> None:
    """Silver quarter: metallic fill, darker rim, and a faint embossed ring."""
    # Rendered from a distance field rather than cv2.circle, which draws
    # filled circles about 1 px larger than requested. Edge pixels are
    # anti-aliased by coverage, so the outer edge sits exactly at `radius`.
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - center[0], yy - center[1])

    def paint(outer: float, inner: float, bgr: tuple[int, int, int]) -> None:
        cov = np.clip(outer - dist + 0.5, 0, 1) * np.clip(dist - inner + 0.5, 0, 1)
        cov = cov[..., None]
        img[:] = (img * (1 - cov) + np.array(bgr, np.float32) * cov).astype(np.uint8)

    rim = max(2.0, radius / 12)
    paint(radius, -1, (175, 175, 180))                   # face
    paint(radius, radius - rim, (95, 95, 100))           # rim
    ring = max(1.0, radius / 20)
    paint(radius * 0.7 + ring / 2, radius * 0.7 - ring / 2, (150, 150, 155))  # embossed ring


def _photo_effects(rng: random.Random, img: np.ndarray, np_rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32) * rng.uniform(0.85, 1.15)          # exposure
    out += np_rng.normal(0, rng.uniform(1, 6), out.shape)            # sensor noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    sigma = rng.uniform(0.3, 1.5)                                    # focus blur
    return cv2.GaussianBlur(out, (0, 0), sigma)


def make_scene(seed: int, skin_tone: str | None = None) -> Scene:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    tone = skin_tone or rng.choice(list(SKIN_TONES))
    px_per_cm = math.exp(rng.uniform(*map(math.log, PX_PER_CM_RANGE)))
    coin_r = COIN_REAL_DIAM_CM / 2 * px_per_cm

    # Target wound area 1–30 cm², capped so it fits beside the coin
    max_r = min(IMG_H, IMG_W) * 0.28
    target_area = math.exp(rng.uniform(math.log(1.0), math.log(30.0)))
    wound_r = min(math.sqrt(target_area / math.pi) * px_per_cm, max_r)

    # Skin with mild texture
    base = np.zeros((IMG_H, IMG_W, 3), np.float32)
    base[:] = SKIN_TONES[tone]
    base += np_rng.normal(0, 4, base.shape)
    skin = np.clip(base, 0, 255).astype(np.uint8)

    wound_center = (int(IMG_W * rng.uniform(0.35, 0.45)), int(IMG_H * rng.uniform(0.4, 0.6)))
    poly = _wound_polygon(rng, wound_center, wound_r)

    wound_mask = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(wound_mask, [poly], 255)
    true_area_cm2 = float(np.count_nonzero(wound_mask)) / px_per_cm ** 2

    # Granulation bed with slough patches inside the wound
    tissue = np.zeros_like(skin)
    tissue[:] = GRANULATION_BGR
    for _ in range(rng.randint(0, 3)):
        c = (wound_center[0] + int(rng.uniform(-0.5, 0.5) * wound_r),
             wound_center[1] + int(rng.uniform(-0.5, 0.5) * wound_r))
        axes = (int(wound_r * rng.uniform(0.15, 0.4)), int(wound_r * rng.uniform(0.15, 0.4)))
        cv2.ellipse(tissue, c, axes, rng.uniform(0, 180), 0, 360, SLOUGH_BGR, -1)
    tissue = np.clip(tissue.astype(np.float32) + np_rng.normal(0, 8, tissue.shape), 0, 255).astype(np.uint8)

    scene = skin.copy()
    scene[wound_mask == 255] = tissue[wound_mask == 255]
    no_coin = scene.copy()

    coin_center = (int(IMG_W * 0.82), int(IMG_H * rng.uniform(0.25, 0.75)))
    _draw_coin(scene, coin_center, coin_r)

    # Same photo effects on both versions so only the coin differs
    effect_seed = rng.randrange(1 << 30)
    with_coin = _photo_effects(random.Random(effect_seed), scene, np.random.default_rng(effect_seed))
    without = _photo_effects(random.Random(effect_seed), no_coin, np.random.default_rng(effect_seed))

    return Scene(with_coin, without, true_area_cm2, px_per_cm, coin_center, coin_r, tone)
