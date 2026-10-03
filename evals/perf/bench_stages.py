"""
bench_stages.py — Time each stage of a wound scan, as the API runs it.

Uses calibration-benchmark scenes, JPEG-encoded like a phone upload, at two
resolutions: the 960×720 benchmark size and a 12 MP phone photo (4032×3024).
The frontend uploads the original file, so 12 MP is what the server sees.

Usage:
    python -m evals.perf.bench_stages [--n 10] [--label baseline]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import vision
from evals.vision.scenes import make_scene

RESULTS_DIR = Path(__file__).parents[1] / "results"
SIZES = {"960x720": (960, 720), "12MP": (4032, 3024)}


def _stage_timer():
    """Wrap vision functions to record how long each takes per call."""
    timings: dict[str, float] = {}
    originals = {}

    def wrap(name: str, label: str):
        fn = getattr(vision, name)
        originals[name] = fn

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            timings[label] = timings.get(label, 0.0) + time.perf_counter() - t0
            return out
        setattr(vision, name, timed)

    for name, label in [
        ("calibrate", "calibrate"),
        ("detect_wound_mask", "wound_mask"),
        ("compute_wound_area", "area"),
        ("ryb_segment", "tissue_kmeans"),
        ("_ryb_cluster", "tissue_kmeans"),
        ("draw_overlay", "overlay_draw"),
    ]:
        if hasattr(vision, name):
            wrap(name, label)

    def restore():
        for name, fn in originals.items():
            setattr(vision, name, fn)
    return timings, restore


def bench(n: int, size: tuple[int, int]) -> dict:
    per_stage: dict[str, list[float]] = {}
    totals: list[float] = []
    for i in range(n):
        scene = make_scene(20_000 + i)
        img = cv2.resize(scene.image, size, interpolation=cv2.INTER_CUBIC)
        ok, upload = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok

        timings, restore = _stage_timer()
        t0 = time.perf_counter()
        t = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(upload.tobytes(), np.uint8), cv2.IMREAD_COLOR)
        timings["decode"] = time.perf_counter() - t
        result = vision.analyze_frame(frame)
        t = time.perf_counter()
        cv2.imencode(".jpg", result["annotated_image"])
        timings["encode"] = time.perf_counter() - t
        total = time.perf_counter() - t0
        restore()

        # Whatever analyze_frame spent outside the named stages
        # (e.g. the second K-Means pass for overlay labels)
        named = sum(v for k, v in timings.items() if k not in ("decode", "encode"))
        analyze_total = total - timings["decode"] - timings["encode"]
        timings["other"] = max(0.0, analyze_total - named)

        totals.append(total)
        for k, v in timings.items():
            per_stage.setdefault(k, []).append(v)

    return {
        "n": n,
        "total_ms": {"p50": round(statistics.median(totals) * 1000, 1),
                     "p95": round(float(np.percentile(totals, 95)) * 1000, 1)},
        "stage_ms_p50": {k: round(statistics.median(v) * 1000, 1) for k, v in per_stage.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()

    vision.analyze_frame(make_scene(0).image)   # warm-up (imports, thread pools)

    results = {"machine": f"{platform.machine()} {platform.processor()}", "sizes": {}}
    for name, size in SIZES.items():
        r = bench(args.n, size)
        results["sizes"][name] = r
        stages = "  ".join(f"{k} {v:.0f}" for k, v in sorted(r["stage_ms_p50"].items(), key=lambda kv: -kv[1]))
        print(f"{name:8} total p50 {r['total_ms']['p50']:.0f} ms  p95 {r['total_ms']['p95']:.0f} ms | {stages}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"perf_stages_{args.label}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
