"""
tests/test_calibration.py — Coin detection and scale calibration tests.

Test categories
---------------
1. detect_coin         — finds a neutral coin, rejects a round red wound
2. refine_coin_radius  — sub-pixel radius on several sizes and skin tones
3. calibrate           — scale from the refined radius, fallback without a coin
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evals.vision.scenes import SKIN_TONES, _draw_coin  # noqa: E402
from vision import (  # noqa: E402
    COIN_REAL_DIAM_CM,
    FALLBACK_CM_PER_PX,
    calibrate,
    detect_coin,
    refine_coin_radius,
)

import cv2  # noqa: E402


def _skin(tone: str = "fair", h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = SKIN_TONES[tone]
    return img


def _with_coin(radius: float, tone: str = "fair", center=(470, 240)) -> np.ndarray:
    img = _skin(tone)
    _draw_coin(img, center, radius)
    return img


# ===========================================================================
# 1. detect_coin
# ===========================================================================

class TestDetectCoin:
    def test_finds_coin_on_skin(self):
        coin = detect_coin(_with_coin(45.0))
        assert coin is not None
        assert abs(coin[0] - 470) <= 3 and abs(coin[1] - 240) <= 3

    def test_round_red_wound_is_not_a_coin(self):
        img = _skin()
        cv2.circle(img, (200, 240), 60, (40, 50, 200), -1)
        assert detect_coin(img) is None

    def test_prefers_coin_over_round_wound(self):
        img = _with_coin(40.0)
        cv2.circle(img, (200, 240), 70, (40, 50, 200), -1)
        coin = detect_coin(img)
        assert coin is not None and abs(coin[0] - 470) <= 3


# ===========================================================================
# 2. refine_coin_radius
# ===========================================================================

class TestRefineCoinRadius:
    @pytest.mark.parametrize("radius", [24.5, 41.3, 77.8])
    def test_subpixel_radius_across_sizes(self, radius):
        img = _with_coin(radius)
        refined = refine_coin_radius(img, detect_coin(img))
        assert refined == pytest.approx(radius, rel=0.02)

    @pytest.mark.parametrize("tone", list(SKIN_TONES))
    def test_radius_on_every_skin_tone(self, tone):
        img = _with_coin(40.0, tone)
        coin = detect_coin(img)
        assert coin is not None
        assert refine_coin_radius(img, coin) == pytest.approx(40.0, rel=0.02)


# ===========================================================================
# 3. calibrate
# ===========================================================================

class TestCalibrate:
    def test_scale_matches_true_resolution(self):
        px_per_cm = 35.0
        img = _with_coin(COIN_REAL_DIAM_CM / 2 * px_per_cm)
        coin, cm_per_px = calibrate(img)
        assert coin is not None
        assert cm_per_px == pytest.approx(1 / px_per_cm, rel=0.02)

    def test_falls_back_without_coin(self):
        coin, cm_per_px = calibrate(_skin())
        assert coin is None
        assert cm_per_px == FALLBACK_CM_PER_PX
