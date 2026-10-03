"""
tests/test_api_auth.py — Token authentication and role-based access.

Uses a temporary copy of the data store with one doctor (D1) assigned to
patient P1, and a second patient P2.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api import auth, main  # noqa: E402
from api.auth import create_token  # noqa: E402

client = TestClient(main.app)


def _bearer(user_id: str, role: str) -> dict:
    return {"Authorization": f"Bearer {create_token(user_id, role)}"}


P1, P2, D1 = _bearer("P1", "patient"), _bearer("P2", "patient"), _bearer("D1", "doctor")


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    def patient(pid: str, email: str) -> dict:
        return {"patient_id": pid, "email": email, "name": pid, "password_hash": main._hash("pw"),
                "wound_history": [{"date": "2026-01-01", "area_cm2": 3.0,
                                   "ryb_ratios": {"red": 90.0, "yellow": 10.0, "black": 0.0}}]}
    patients, doctors = tmp_path / "patients.json", tmp_path / "doctors.json"
    patients.write_text(json.dumps([patient("P1", "p1@x.com"), patient("P2", "p2@x.com")]))
    doctors.write_text(json.dumps([{"doctor_id": "D1", "email": "d1@x.com", "name": "Dr One",
                                    "specialty": "Wound care", "password_hash": main._hash("pw"),
                                    "patient_ids": ["P1"]}]))
    monkeypatch.setattr(main, "_PATIENTS_PATH", str(patients))
    monkeypatch.setattr(main, "_DOCTORS_PATH", str(doctors))


# ===========================================================================
# Tokens
# ===========================================================================

class TestTokens:
    def test_login_returns_working_token(self):
        r = client.post("/api/auth/login", json={"email": "p1@x.com", "password": "pw"})
        token = r.json()["token"]
        assert client.get("/api/patients/P1", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    def test_registration_returns_token(self):
        r = client.post("/api/auth/register/patient",
                        json={"name": "New", "email": "new@x.com", "password": "pw", "age": 40})
        assert r.status_code == 200 and r.json()["token"]

    @pytest.mark.parametrize("headers", [
        {},
        {"Authorization": "Bearer not-a-token"},
        {"Authorization": "Basic abc"},
    ])
    def test_missing_or_bad_token_is_401(self, headers):
        assert client.get("/api/patients/P1", headers=headers).status_code == 401

    def test_expired_token_is_401(self, monkeypatch):
        monkeypatch.setattr(auth, "TOKEN_TTL", timedelta(seconds=-1))
        expired = {"Authorization": f"Bearer {create_token('P1', 'patient')}"}
        assert client.get("/api/patients/P1", headers=expired).status_code == 401


# ===========================================================================
# Access rules
# ===========================================================================

@pytest.mark.parametrize("method,path,headers,expected", [
    # Patients: own record only
    ("get",    "/api/patients/P1",               P1, 200),
    ("get",    "/api/patients/P2",               P1, 403),
    ("get",    "/api/patients/P1/analysis",      P1, 200),
    ("get",    "/api/patients/P2/analysis",      P1, 403),
    ("get",    "/api/patients",                  P1, 403),
    ("get",    "/api/doctors/D1",                P1, 403),
    ("get",    "/api/doctors/D1/patients",       P1, 403),
    ("post",   "/api/doctors/D1/patients/P2",    P1, 403),
    # Doctors: any patient, own doctor resources only
    ("get",    "/api/patients",                  D1, 200),
    ("get",    "/api/patients/P2",               D1, 200),
    ("get",    "/api/doctors/D1",                D1, 200),
    ("get",    "/api/doctors/D1/patients",       D1, 200),
    ("post",   "/api/doctors/D1/patients/P2",    D1, 200),
    ("delete", "/api/doctors/D1/patients/P1",    D1, 200),
    ("get",    "/api/doctors/D2",                D1, 403),
    # Unauthenticated
    ("get",    "/api/patients",                  {}, 401),
    ("get",    "/api/doctors/D1/patients",       {}, 401),
])
def test_access_rules(method, path, headers, expected):
    assert getattr(client, method)(path, headers=headers).status_code == expected


def test_patient_cannot_save_scan_to_another_patient():
    body = {"area_cm2": 2.0, "ryb_ratios": {"red": 90.0, "yellow": 10.0, "black": 0.0}}
    assert client.post("/api/patients/P2/scan", json=body, headers=P1).status_code == 403
    assert client.post("/api/patients/P1/scan", json=body, headers=P1).status_code == 200


def test_patient_cannot_scan_as_another_patient():
    r = client.post("/api/scan/analyze", data={"patient_id": "P2"}, headers=P1)
    assert r.status_code == 403


def test_public_doctor_directory_hides_private_fields():
    r = client.get("/api/doctors")
    assert r.status_code == 200
    assert set(r.json()[0]) == {"doctor_id", "name", "specialty"}


def test_no_endpoint_returns_password_hashes():
    for path, headers in [("/api/patients", D1), ("/api/patients/P1", P1), ("/api/doctors/D1", D1)]:
        assert "password_hash" not in client.get(path, headers=headers).text
