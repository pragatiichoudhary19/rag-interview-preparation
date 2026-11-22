
"""Thin typed wrapper over the FastAPI backend."""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st


def get_backend_url() -> str:
    # Streamlit Cloud: read from App Settings > Secrets
    try:
        url = st.secrets["BACKEND_URL"]
    except (KeyError, FileNotFoundError):
        # Local development fallback
        url = os.getenv("BACKEND_URL", "http://localhost:8000")

    return str(url).rstrip("/")


BACKEND_URL = get_backend_url()


class APIError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class APIClient:
    def __init__(self, token: str | None = None, base_url: str = BACKEND_URL):
        self.token, self.base = token, base_url

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, timeout: float = 180, **kw: Any) -> Any:
        try:
            r = httpx.request(
                method,
                self.base + path,
                headers=self.headers,
                timeout=timeout,
                **kw
            )
        except httpx.HTTPError as exc:
            raise APIError(
                0, f"Cannot reach the backend at {self.base}: {exc}"
            ) from exc

        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text

            if isinstance(detail, list):
                detail = "; ".join(
                    f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}"
                    for d in detail
                )

            raise APIError(r.status_code, str(detail))

        return r.json() if r.content else None

    get = lambda self, p, **kw: self._request("GET", p, **kw)
    post = lambda self, p, **kw: self._request("POST", p, **kw)
    put = lambda self, p, **kw: self._request("PUT", p, **kw)
    delete = lambda self, p, **kw: self._request("DELETE", p, **kw)

    # Authentication
    def register(self, email: str, full_name: str, password: str):
        return self.post(
            "/auth/register",
            json={
                "email": email,
                "full_name": full_name,
                "password": password
            }
        )

    def login(self, email: str, password: str):
        return self.post(
            "/auth/login",
            json={"email": email, "password": password}
        )

    # Content
    def upload(self, path: str, filename: str, data: bytes, mime: str, form: dict | None = None):
        return self.post(
            path,
            files={"file": (filename, data, mime)},
            data=form or {}
        )

    # Interviews
    def start_interview(self, **cfg):
        return self.post("/interviews/start", json={**cfg, "stream": True})

    def interview(self, iid: str):
        return self.get(f"/interviews/{iid}")

    def answer(self, iid: str, answer: str, extra: str | None = None):
        return self.post(
            f"/interviews/{iid}/answer",
            json={
                "answer": answer,
                "additional_context": extra,
                "stream": True
            }
        )

    def decide(self, iid: str, decision: str, **kw):
        return self.post(
            f"/interviews/{iid}/next",
            json={"decision": decision, "stream": True, **kw}
        )

    def end(self, iid: str):
        return self.post("/interviews/{iid}/end", json={"stream": True})
