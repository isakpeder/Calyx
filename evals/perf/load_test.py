"""
load_test.py — Concurrent-user load test of the scan endpoint.

Starts the API exactly as Railway does (one uvicorn worker), then simulated
users each upload 12 MP wound photos to POST /api/scan/analyze back to back.
Reports end-to-end latency (client-side, includes upload) and throughput at
each concurrency level. The endpoint does not write to the data store.

Usage:
    python -m evals.perf.load_test [--users 1 10 25 50] [--requests-per-user 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import httpx
import numpy as np

from evals.vision.scenes import make_scene

RESULTS_DIR = Path(__file__).parents[1] / "results"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
PATIENT_ID = "P001"
N_IMAGES = 8

# Shared with the server subprocess so this script can mint a valid token
os.environ.setdefault("CALYX_JWT_SECRET", "calyx-load-test-secret-not-for-production")
from api.auth import create_token  # noqa: E402

AUTH = {"Authorization": f"Bearer {create_token(PATIENT_ID, 'patient')}"}


def _uploads() -> list[bytes]:
    out = []
    for i in range(N_IMAGES):
        img = cv2.resize(make_scene(30_000 + i).image, (4032, 3024), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        out.append(buf.tobytes())
    return out


def _start_server(workers: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
           "--port", str(PORT), "--workers", str(workers), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, cwd=Path(__file__).parents[2], env=os.environ.copy())
    for _ in range(100):
        try:
            if httpx.get(f"{BASE}/api/doctors", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not start")


async def _user(client: httpx.AsyncClient, uploads: list[bytes], n: int, offset: int,
                latencies: list[float], errors: list[str]) -> None:
    for k in range(n):
        body = uploads[(offset + k) % len(uploads)]
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{BASE}/api/scan/analyze",
                                  data={"patient_id": PATIENT_ID},
                                  files={"file": ("wound.jpg", body, "image/jpeg")},
                                  headers=AUTH)
            if r.status_code != 200:
                errors.append(str(r.status_code))
                continue
        except httpx.HTTPError as exc:
            errors.append(type(exc).__name__)
            continue
        latencies.append(time.perf_counter() - t0)


async def _level(uploads: list[bytes], users: int, per_user: int) -> dict:
    latencies: list[float] = []
    errors: list[str] = []
    limits = httpx.Limits(max_connections=users)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(_user(client, uploads, per_user, u, latencies, errors) for u in range(users)))
        wall = time.perf_counter() - t0
    return {
        "users": users,
        "requests": len(latencies) + len(errors),
        "errors": len(errors),
        "error_types": dict(Counter(errors)),
        "p50_ms": round(statistics.median(latencies) * 1000) if latencies else None,
        "p95_ms": round(float(np.percentile(latencies, 95)) * 1000) if latencies else None,
        "throughput_rps": round(len(latencies) / wall, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, nargs="+", default=[1, 10, 25, 50])
    parser.add_argument("--requests-per-user", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--label", default="current")
    args = parser.parse_args()

    uploads = _uploads()
    print(f"upload size ~{statistics.mean(map(len, uploads)) / 1e6:.1f} MB (12 MP JPEG)")

    proc = _start_server(args.workers)
    try:
        asyncio.run(_level(uploads, 2, 1))  # warm-up
        levels = []
        for users in args.users:
            r = asyncio.run(_level(uploads, users, args.requests_per_user))
            levels.append(r)
            print(f"{users:3} users: p50 {r['p50_ms']} ms  p95 {r['p95_ms']} ms  "
                  f"{r['throughput_rps']} req/s  errors {r['errors']}/{r['requests']} {r['error_types'] or ''}")
    finally:
        proc.terminate()
        proc.wait()

    results = {
        "machine": f"{platform.machine()} {platform.processor()}, {os.cpu_count()} cores",
        "workers": args.workers,
        "levels": levels,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"perf_load_{args.label}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
