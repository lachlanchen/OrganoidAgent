"""Metadata-only persistence for human review decisions."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

REVIEW_STATUSES = {"pending", "approved", "rejected", "needs_correction"}
REVIEW_FLAGS = {"debris", "edge_clipping", "bad_focus", "wrong_channel", "bad_pairing", "low_confidence", "mask_error", "other"}


def _token(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.:-]", "", str(value or "").strip())
    return text or fallback


class ReviewStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, review_id: str) -> Path:
        return self.root / f"{_token(review_id, 'invalid')}.json"

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def list(self, job_id: str = "", target_key: str = "") -> list[dict[str, Any]]:
        records = []
        for path in self.root.glob("review-*.json"):
            record = self._read(path)
            if record and (not job_id or record.get("job_id") == job_id) and (not target_key or record.get("target_key") == target_key):
                records.append(record)
        return sorted(records, key=lambda item: float(item.get("updated_at", 0)), reverse=True)

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("review payload must be an object")
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("job_id is required")
        status = str(payload.get("status") or "pending").strip().lower()
        if status not in REVIEW_STATUSES:
            raise ValueError(f"unsupported review status: {status}")
        raw_flags = payload.get("flags") or []
        if not isinstance(raw_flags, list):
            raise ValueError("flags must be a list")
        review_id = _token(payload.get("review_id"), "")
        existing = self._read(self._path(review_id)) if review_id else None
        if existing is None:
            review_id = f"review-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
            created_at = time.time()
        else:
            created_at = float(existing.get("created_at", time.time()))
        record = {
            "review_id": review_id, "job_id": job_id,
            "target_label": str(payload.get("target_label") or job_id).strip()[:240],
            "target_key": str(payload.get("target_key") or "").strip()[:1000],
            "target_path": str(payload.get("target_path") or "").strip()[:2000],
            "status": status, "reviewer": str(payload.get("reviewer") or "").strip()[:160],
            "notes": str(payload.get("notes") or "").strip()[:4000],
            "flags": sorted({str(flag).strip() for flag in raw_flags if str(flag).strip()} & REVIEW_FLAGS),
            "created_at": created_at, "updated_at": time.time(),
        }
        path = self._path(review_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return record
