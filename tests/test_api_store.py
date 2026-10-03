"""
tests/test_api_store.py — JSON data store under concurrent writes.

Endpoints are called from many threads at once (FastAPI runs sync
endpoints in a thread pool) against a temporary copy of the store.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api import main  # noqa: E402


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    patients, doctors = tmp_path / "patients.json", tmp_path / "doctors.json"
    patients.write_text(json.dumps([{"patient_id": "P1", "email": "p1@x.com", "wound_history": []}]))
    doctors.write_text(json.dumps([{"doctor_id": "D1", "email": "d1@x.com", "patient_ids": []}]))
    monkeypatch.setattr(main, "_PATIENTS_PATH", str(patients))
    monkeypatch.setattr(main, "_DOCTORS_PATH", str(doctors))
    return patients, doctors


def _register(i: int) -> None:
    main.register_patient(main.RegisterPatientRequest(
        name=f"Patient {i}", email=f"user{i}@x.com", password="pw", age=50,
    ))


def test_concurrent_registrations_are_all_saved(temp_store):
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_register, range(200)))
    assert len(main._load_patients()) == 201


def test_concurrent_scans_are_all_saved(temp_store):
    req = main.ScanRequest(area_cm2=3.0, ryb_ratios={"red": 80.0, "yellow": 20.0, "black": 0.0})
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: main.add_scan("P1", req), range(100)))
    assert len(main._load_patients()[0]["wound_history"]) == 100


def test_concurrent_doctor_assignments_are_all_saved(temp_store):
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_register, range(50)))
    ids = [p["patient_id"] for p in main._load_patients()]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda pid: main.add_patient_to_doctor("D1", pid), ids))
    assert sorted(main._load_doctors()[0]["patient_ids"]) == sorted(ids)


def test_locked_endpoint_still_parses_request_body(temp_store):
    from fastapi.testclient import TestClient
    r = TestClient(main.app).post("/api/auth/register/patient", json={
        "name": "Via HTTP", "email": "http@x.com", "password": "pw", "age": 40,
    })
    assert r.status_code == 200 and r.json()["name"] == "Via HTTP"
