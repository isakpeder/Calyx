"""
auth.py — Bearer-token authentication and role-based access for the API.

Login and registration issue a signed JWT carrying the user's ID and role.
Route dependencies then enforce:
  * patients may only read or modify their own record
  * doctors may read any patient (to search and add them) but only manage
    their own doctor profile and patient list
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, Header, HTTPException

TOKEN_TTL = timedelta(hours=12)
_ALGORITHM = "HS256"

_SECRET = os.environ.get("CALYX_JWT_SECRET")
if not _SECRET:
    # Dev fallback: tokens stop working when the server restarts.
    _SECRET = secrets.token_urlsafe(32)
    logging.getLogger(__name__).warning("CALYX_JWT_SECRET not set; using a random per-process secret")


def create_token(user_id: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "role": role, "iat": now, "exp": now + TOKEN_TTL}
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def current_user(authorization: str | None = Header(None)) -> dict:
    """Decode the Bearer token; 401 if missing, malformed or expired."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        claims = jwt.decode(authorization.removeprefix("Bearer "), _SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"id": claims["sub"], "role": claims["role"]}


def check_patient_access(user: dict, patient_id: str) -> None:
    """Doctors may access any patient; patients only themselves."""
    if user["role"] == "doctor":
        return
    if user["role"] == "patient" and user["id"] == patient_id:
        return
    raise HTTPException(status_code=403, detail="Not allowed to access this patient")


def patient_access(patient_id: str, user: dict = Depends(current_user)) -> None:
    check_patient_access(user, patient_id)


def doctor_only(user: dict = Depends(current_user)) -> None:
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only")


def doctor_self(doctor_id: str, user: dict = Depends(current_user)) -> None:
    if user["role"] != "doctor" or user["id"] != doctor_id:
        raise HTTPException(status_code=403, detail="Not allowed to access this doctor")
