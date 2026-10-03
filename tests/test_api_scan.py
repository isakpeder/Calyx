"""
tests/test_api_scan.py — POST /api/scan/analyze upload handling.
"""

from __future__ import annotations

import base64
import os
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api import main  # noqa: E402
from api.auth import create_token  # noqa: E402

client = TestClient(main.app)
AUTH = {"Authorization": f"Bearer {create_token('P001', 'patient')}"}


def _jpeg(h: int = 400, w: int = 400) -> bytes:
    img = np.full((h, w, 3), (110, 130, 180), np.uint8)
    cv2.circle(img, (w // 2, h // 2), 80, (40, 50, 200), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _post(body: bytes):
    return client.post("/api/scan/analyze", data={"patient_id": "P001"},
                       files={"file": ("wound.jpg", body, "image/jpeg")}, headers=AUTH)


def test_valid_upload_returns_assessment():
    r = _post(_jpeg())
    assert r.status_code == 200
    assert r.json()["area_cm2"] > 0 and "priority" in r.json()


def test_unreadable_image_is_400():
    assert _post(b"not an image").status_code == 400


def test_oversized_upload_is_413(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    assert _post(_jpeg()).status_code == 413


def test_12mp_photo_is_downscaled():
    r = _post(_jpeg(3024, 4032))
    assert r.status_code == 200
    annotated = base64.b64decode(r.json()["annotated_image_b64"])
    img = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 1280
