#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_STATUSES = {"succeeded", "failed", "cancelled"}
REASONING_LEVELS = {"low", "medium", "high", "xhigh"}
ALLOWED_STEP_BLOCKS = {"plan", "work", "debug", "fix", "summary", "commit_push"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def safe_token(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(value or "")):
        raise ValueError(f"invalid {label}")
    return str(value)


def normalize_reasoning(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip().lower()
    return text if text in REASONING_LEVELS else fallback


@dataclass
class ParseError(Exception):
    message: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"error": "parse_error", "message": self.message, "line": self.line}


def _strip_comment(line: str) -> str:
    in_string = False
    escaped = False
    for idx, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if char == "#" and not in_string:
            return line[:idx]
    return line


def _parse_json_object(raw: str, line_no: int) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON object: {exc.msg}", line_no) from exc
    if not isinstance(payload, dict):
        raise ParseError("statement payload must be a JSON object", line_no)
    return payload


def parse_aaps(text: str) -> dict[str, Any]:
    """Parse the compact AAPS pipeline format used by AutoAppDev-style studios."""
    lines = text.splitlines()
    first = next((line.strip() for line in lines if line.strip() and not line.strip().startswith("#")), "")
    if first != "AUTOAPPDEV_PIPELINE 1":
        raise ParseError('first non-comment line must be "AUTOAPPDEV_PIPELINE 1"', 1)

    tasks: list[dict[str, Any]] = []
    current_task: dict[str, Any] | None = None
    current_step: dict[str, Any] | None = None

    for line_no, raw in enumerate(lines, start=1):
        line = _strip_comment(raw).strip()
        if not line or line == "AUTOAPPDEV_PIPELINE 1":
            continue
        keyword, sep, rest = line.partition(" ")
        if not sep:
            raise ParseError("expected KEYWORD followed by JSON object", line_no)
        payload = _parse_json_object(rest.strip(), line_no)

        if keyword == "TASK":
            task = {
                "id": str(payload.get("id") or f"task_{len(tasks) + 1}"),
                "title": str(payload.get("title") or "Untitled task"),
                "objective": str(payload.get("objective") or ""),
                "metadata": {k: v for k, v in payload.items() if k not in {"id", "title", "objective"}},
                "steps": [],
            }
            tasks.append(task)
            current_task = task
            current_step = None
        elif keyword == "STEP":
            if current_task is None:
                raise ParseError("STEP requires a preceding TASK", line_no)
            block = str(payload.get("block") or "work").strip()
            if block not in ALLOWED_STEP_BLOCKS:
                raise ParseError(f"unsupported step block: {block}", line_no)
            step = {
                "id": str(payload.get("id") or f"step_{len(current_task['steps']) + 1}"),
                "block": block,
                "title": str(payload.get("title") or block.replace("_", " ").title()),
                "instruction": str(payload.get("instruction") or ""),
                "metadata": {k: v for k, v in payload.items() if k not in {"id", "block", "title", "instruction"}},
                "actions": [],
            }
            current_task["steps"].append(step)
            current_step = step
        elif keyword == "ACTION":
            if current_step is None:
                raise ParseError("ACTION requires a preceding STEP", line_no)
            current_step["actions"].append(payload)
        else:
            raise ParseError(f"unknown statement: {keyword}", line_no)

    if not tasks:
        raise ParseError("pipeline must contain at least one TASK", None)
    return {"kind": "autoappdev_ir", "version": 1, "tasks": tasks}


class StudioChatStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def new_session(self, title: str = "OrganoidAgent chat") -> dict[str, Any]:
        session_id = "chat-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        session_dir = self.root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        session = {"id": session_id, "title": title, "created_at": now_iso(), "updated_at": now_iso()}
        atomic_write_json(session_dir / "session.json", session)
        atomic_write_text(session_dir / "messages.jsonl", "")
        return session

    def session_dir(self, session_id: str) -> Path:
        return self.root / safe_token(session_id, "session id")

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = read_json(self.session_dir(session_id) / "session.json")
        if not isinstance(session, dict):
            raise FileNotFoundError(session_id)
        return session

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        path = self.session_dir(session_id) / "messages.jsonl"
        messages = []
        if not path.exists():
            return messages
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    msg = None
                if isinstance(msg, dict):
                    messages.append(msg)
        return messages

    def append_message(self, session_id: str, role: str, content: str, **extra: Any) -> dict[str, Any]:
        session_dir = self.session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        message = {
            "id": "msg-" + uuid.uuid4().hex[:12],
            "role": role,
            "content": content,
            "created_at": now_iso(),
            **extra,
        }
        with (session_dir / "messages.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(message, ensure_ascii=False) + "\n")
        session = self.get_session(session_id)
        session["updated_at"] = now_iso()
        atomic_write_json(session_dir / "session.json", session)
        return message


class CodexJobError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class CodexJobManager:
    def __init__(self, repo_root: Path, runtime_root: Path):
        self.repo_root = repo_root
        self.runtime_root = runtime_root
        self.jobs_root = runtime_root / "codex-jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.default_model = os.environ.get("ORGANOID_AGENT_CODEX_MODEL", "gpt-5.5")
        self.default_timeout_s = float(os.environ.get("ORGANOID_AGENT_CODEX_TIMEOUT_S", "1200"))
        self.mock = os.environ.get("ORGANOID_AGENT_MOCK_CODEX", "").lower() in {"1", "true", "yes", "on"}
        self._tasks: dict[str, asyncio.Task] = {}

    def new_job_id(self, tool: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{tool}-{stamp}-{uuid.uuid4().hex[:8]}"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / safe_token(job_id, "job id")

    def read_job(self, job_id: str) -> dict[str, Any]:
        job = read_json(self.job_dir(job_id) / "job.json")
        if not isinstance(job, dict):
            raise FileNotFoundError(job_id)
        return job

    def write_job(self, job: dict[str, Any]) -> dict[str, Any]:
        atomic_write_json(self.job_dir(str(job["id"])) / "job.json", job)
        return job

    def update_job(self, job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        job = self.read_job(job_id)
        job.update(updates)
        job["updated_at"] = now_iso()
        return self.write_job(job)

    def normalize_tool(self, raw: Any) -> str:
        tool = str(raw or "response").strip().lower()
        if tool == "respond":
            tool = "response"
        if tool not in {"response", "assistant"}:
            raise CodexJobError("invalid_tool", "tool must be response or assistant")
        return tool

    def default_reasoning_for_tool(self, tool: str) -> str:
        return "high" if tool == "assistant" else "medium"

    def build_prompt(self, job: dict[str, Any], payload: dict[str, Any]) -> str:
        mode = "assistant job that may inspect and edit this repo" if job["tool"] == "assistant" else "reply job"
        transcript = payload.get("transcript")
        pipeline = str(payload.get("pipeline_text") or "").strip()
        extra_context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        parts = [
            "You are OrganoidAgent Studio running inside the local OrganoidAgent repository.",
            f"Mode: {mode}.",
            "Keep answers concise and concrete. If this is an assistant job, complete the requested repo task end-to-end when feasible.",
            "",
            "User request:",
            str(payload.get("prompt") or "").strip(),
        ]
        if transcript:
            parts.extend(["", "Recent chat transcript JSON:", json.dumps(transcript, ensure_ascii=False, indent=2)])
        if pipeline:
            parts.extend(["", "Current AAPS pipeline script:", pipeline])
        if extra_context:
            parts.extend(["", "Additional context JSON:", json.dumps(extra_context, ensure_ascii=False, indent=2)])
        return "\n".join(parts).strip() + "\n"

    def submit_job(self, payload: dict[str, Any], *, start: bool = True) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CodexJobError("invalid_body", "request body must be a JSON object")
        tool = self.normalize_tool(payload.get("tool"))
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise CodexJobError("empty_prompt", "prompt is required")
        reasoning = normalize_reasoning(payload.get("reasoning"), self.default_reasoning_for_tool(tool))
        allow_edits = bool(payload.get("allow_edits", tool == "assistant"))
        model = str(payload.get("model") or self.default_model).strip() or self.default_model

        job_id = self.new_job_id(tool)
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_id,
            "tool": tool,
            "status": "queued",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": None,
            "model": model,
            "reasoning": reasoning,
            "allow_edits": allow_edits,
            "session_id": str(payload.get("session_id") or ""),
            "prompt_preview": prompt[:240],
            "poll_url": f"/api/agent/codex/job?id={job_id}",
            "result_url": f"/api/agent/codex/result?id={job_id}",
            "paths": {
                "dir": str(job_dir),
                "input": str(job_dir / "input.json"),
                "prompt": str(job_dir / "prompt.txt"),
                "output": str(job_dir / "output.txt"),
                "stdout": str(job_dir / "stdout.log"),
                "stderr": str(job_dir / "stderr.log"),
            },
        }
        atomic_write_json(job_dir / "input.json", payload)
        self.write_job(job)
        if start:
            self.start_job(job_id)
        return self.job_status(job_id, include_logs=False, include_output=False)

    def start_job(self, job_id: str) -> None:
        jid = safe_token(job_id, "job id")
        task = self._tasks.get(jid)
        if task and not task.done():
            return
        self._tasks[jid] = asyncio.create_task(self.run_job(jid))

    async def run_job(self, job_id: str) -> dict[str, Any]:
        started = time.monotonic()
        job_dir = self.job_dir(job_id)
        payload = read_json(job_dir / "input.json", {})
        if not isinstance(payload, dict):
            payload = {}
        try:
            job = self.update_job(job_id, {"status": "running", "started_at": now_iso()})
            prompt = self.build_prompt(job, payload)
            atomic_write_text(job_dir / "prompt.txt", prompt)

            if self.mock or bool(payload.get("mock")):
                atomic_write_text(job_dir / "output.txt", f"Mock OrganoidAgent Studio result:\n{payload.get('prompt', '')}\n")
                return self.update_job(job_id, {"status": "succeeded", "finished_at": now_iso(), "elapsed_seconds": round(time.monotonic() - started, 2), "returncode": 0})
            if not shutil.which("codex"):
                raise CodexJobError("codex_not_found", "codex executable was not found on PATH")

            cmd = [
                "codex",
                "exec",
                "--ephemeral",
                "--model",
                str(job["model"]),
                "-c",
                f'model_reasoning_effort="{job["reasoning"]}"',
                "--cd",
                str(self.repo_root),
                "--output-last-message",
                str(job_dir / "output.txt"),
            ]
            if bool(job.get("allow_edits")):
                cmd.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                cmd.extend(["--sandbox", "read-only"])
            cmd.append("-")

            timeout_s = max(10.0, min(float(payload.get("timeout_s") or self.default_timeout_s), 7200.0))
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            try:
                out_b, err_b = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise CodexJobError("timeout", f"codex exec exceeded timeout_s={timeout_s}") from exc
            atomic_write_text(job_dir / "stdout.log", out_b.decode("utf-8", errors="replace"))
            atomic_write_text(job_dir / "stderr.log", err_b.decode("utf-8", errors="replace"))
            status = "succeeded" if proc.returncode == 0 and (job_dir / "output.txt").exists() else "failed"
            updates = {"status": status, "finished_at": now_iso(), "elapsed_seconds": round(time.monotonic() - started, 2), "returncode": proc.returncode}
            if status == "failed":
                updates["error"] = f"codex exec failed with returncode {proc.returncode}"
            return self.update_job(job_id, updates)
        except CodexJobError as exc:
            atomic_write_text(job_dir / "stderr.log", exc.detail + "\n")
            return self.update_job(job_id, {"status": "failed", "finished_at": now_iso(), "elapsed_seconds": round(time.monotonic() - started, 2), "error": exc.code, "detail": exc.detail})
        except Exception as exc:
            atomic_write_text(job_dir / "stderr.log", f"{type(exc).__name__}: {exc}\n")
            return self.update_job(job_id, {"status": "failed", "finished_at": now_iso(), "elapsed_seconds": round(time.monotonic() - started, 2), "error": f"{type(exc).__name__}: {exc}"})

    def job_status(self, job_id: str, *, include_logs: bool = True, include_output: bool = True) -> dict[str, Any]:
        job = self.read_job(job_id)
        job_dir = self.job_dir(str(job["id"]))
        payload: dict[str, Any] = {"job": job}
        if include_output and (job_dir / "output.txt").exists():
            payload["output_text"] = (job_dir / "output.txt").read_text(encoding="utf-8", errors="replace")
        if include_logs:
            payload["logs"] = {
                "stdout_tail": tail_text(job_dir / "stdout.log"),
                "stderr_tail": tail_text(job_dir / "stderr.log"),
            }
        return payload

    def list_jobs(self, limit: int = 20, session_id: str | None = None) -> list[dict[str, Any]]:
        jobs = []
        for path in self.jobs_root.glob("*/job.json"):
            job = read_json(path)
            if not isinstance(job, dict):
                continue
            if session_id and str(job.get("session_id") or "") != session_id:
                continue
            jobs.append(job)
        return sorted(jobs, key=lambda item: str(item.get("created_at") or ""), reverse=True)[: max(1, min(limit, 200))]
