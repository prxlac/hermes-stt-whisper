"""Run mlx-whisper on a Mac over SSH (the fast path).

A webservice backend running CPU-only is ~0.7x realtime; a Mac with mlx-whisper (Apple
Silicon, Metal) measured ~28x realtime on Croatian audio. This module shells out to
``ssh``/``scp`` and never raises: every failure becomes
``{"success": False, "error": ...}`` so the router can fall back to the webservice.

Why a script ships with the plugin instead of reusing a helper left on the Mac: that
kind of file hardcodes language, model and output naming, and it is user state nobody
versions. ``remote/mlx_transcribe.py`` is scp'd per run and takes its arguments.

Settings (``stt.whisper-asr.mac`` in config.yaml)::

    mac:
      enabled: true
      host: mac-stt                  # ssh alias for the Mac
      venv_python: ~/mlx-asr/venv/bin/python
      model: mlx-community/whisper-large-v3-turbo
      probe_timeout: 2
      down_ttl: 120
      timeout: 300
      ffmpeg: ""                     # "" -> PATH lookup, then ~/.hermes/tools/ffmpeg-*
"""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import whisper_asr_core as core

REMOTE_SCRIPT = Path(__file__).resolve().parent / "remote" / "mlx_transcribe.py"

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "host": "mac-stt",
    "venv_python": "~/mlx-asr/venv/bin/python",
    "model": "mlx-community/whisper-large-v3-turbo",
    "probe_timeout": 2.0,
    "down_ttl": 120.0,
    "timeout": 300.0,
    "ffmpeg": "",
}


def settings_from(config_block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a user config block over the defaults, coercing types defensively."""
    settings = dict(DEFAULTS)
    block = config_block if isinstance(config_block, dict) else {}
    for key in ("host", "venv_python", "model", "ffmpeg"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            settings[key] = value.strip()
    for key in ("probe_timeout", "down_ttl", "timeout"):
        raw = block.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            settings[key] = value
    if "enabled" in block:
        raw = block.get("enabled")
        settings["enabled"] = (
            raw if isinstance(raw, bool) else str(raw).strip().lower() in {"1", "true", "yes", "on"}
        )
    return settings


def _remote(path: str) -> str:
    """Quote a path for the remote shell, expanding a leading ``~`` there.

    ``shlex.quote('~/x')`` produces a *literal* tilde, which the remote shell will not
    expand — that silently breaks the venv probe. ``$HOME`` inside double quotes does
    expand remotely.
    """
    text = str(path).strip()
    if text.startswith("~/"):
        return '"$HOME/' + text[2:].replace('"', '\\"') + '"'
    return shlex.quote(text)


def _run(cmd: List[str], timeout: float) -> Tuple[int, str, str]:
    """Run a command, folding "not installed"/timeout into an exit code + message."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not installed"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]} timed out after {timeout:.0f}s"
    return proc.returncode, proc.stdout, proc.stderr


def _down_marker(host: str) -> Path:
    import tempfile  # noqa: PLC0415 — keep module import cost low

    return Path(tempfile.gettempdir()) / f"hermes-asr-down-{host}.stamp"


def probe(settings: Dict[str, Any]) -> Tuple[bool, str]:
    """Is the Mac reachable *and* is its mlx venv still there? One SSH round-trip.

    A failed probe is remembered for ``down_ttl`` seconds: without that, every voice
    message while the Mac sleeps pays the full ConnectTimeout again before falling back.
    """
    if not settings.get("enabled", True):
        return False, "disabled in config"
    host = settings["host"]
    marker = _down_marker(host)
    ttl = float(settings.get("down_ttl") or 0)
    try:
        if ttl > 0 and marker.exists() and (time.time() - marker.stat().st_mtime) < ttl:
            return False, f"skipped: {host} failed a probe less than {int(ttl)}s ago"
    except OSError:
        pass

    rc, out, err = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(settings['probe_timeout'])}",
            host,
            f"test -x {_remote(settings['venv_python'])} && echo HERMES_ASR_OK",
        ],
        timeout=settings["probe_timeout"] + 3,
    )
    if rc == 0 and "HERMES_ASR_OK" in out:
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
        return True, "ok"
    try:
        marker.touch()
    except OSError:
        pass
    detail = " ".join((err or out or "").split())[:200]
    if rc == 0:
        detail = f"venv missing at {settings['venv_python']}"
    return False, detail or f"ssh {host} failed (exit {rc})"


def transcribe(
    file_path: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe *file_path* on the Mac. Never raises."""
    settings = settings_from(settings)
    if not REMOTE_SCRIPT.exists():
        return {"success": False, "transcript": "", "error": f"missing {REMOTE_SCRIPT}"}

    prepared: Dict[str, Any] = {}
    host = settings["host"]
    tag = uuid.uuid4().hex[:12]
    remote_wav = f"/tmp/hermes-asr-{tag}.wav"
    # scp keeps the source basename; the content is identical every run so a
    # shared remote name is safe even with two concurrent transcriptions.
    remote_script = f"/tmp/{REMOTE_SCRIPT.name}"
    remote_base = f"/tmp/hermes-asr-{tag}"
    staged = False

    def cleanup_remote() -> None:
        if not staged:
            return
        _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
             f"rm -f {shlex.quote(remote_wav)} {shlex.quote(remote_base)}.txt "
             f"{shlex.quote(remote_base)}.jsonl {shlex.quote(remote_script)}"],
            timeout=15,
        )

    try:
        prepared = core.prepare_wav16k(
            file_path, ffmpeg=settings["ffmpeg"], timeout=min(settings["timeout"], 120)
        )
        if prepared.get("error"):
            return {"success": False, "transcript": "", "error": prepared["error"]}
        wav_path = Path(prepared["path"])

        rc, _, err = _run(
            ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             str(wav_path), str(REMOTE_SCRIPT), f"{host}:/tmp/"],
            timeout=min(settings["timeout"], 300),
        )
        if rc != 0:
            return {"success": False, "transcript": "", "error": f"scp to {host} failed: {' '.join(err.split())[:200]}"}
        staged = True

        # scp lands both files under their own names; give the wav the unique path.
        rc, _, err = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
             f"mv -f /tmp/{wav_path.name} {shlex.quote(remote_wav)}"],
            timeout=30,
        )
        if rc != 0:
            return {"success": False, "transcript": "", "error": f"remote staging failed: {' '.join(err.split())[:200]}"}

        script_model = (model or settings["model"]).strip()
        lang = (language or "").strip()
        remote_cmd = (
            f"{_remote(settings['venv_python'])} {shlex.quote(remote_script)} "
            f"{shlex.quote(remote_wav)} {shlex.quote(remote_base)} "
            f"{shlex.quote(lang or 'auto')} {shlex.quote(script_model)}"
        )
        rc, out, err = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(settings['probe_timeout'])}", host, remote_cmd],
            timeout=settings["timeout"],
        )
        if rc != 0:
            detail = " ".join((err or out or "").split())[-400:]
            return {"success": False, "transcript": "", "error": f"mlx_whisper on {host} failed (exit {rc}): {detail}"}

        rc, txt, err = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
             f"cat {shlex.quote(remote_base)}.txt"],
            timeout=60,
        )
        if rc != 0:
            return {"success": False, "transcript": "", "error": f"reading transcript from {host} failed: {' '.join(err.split())[:200]}"}

        transcript = " ".join(line.strip() for line in txt.splitlines() if line.strip())
        return {"success": True, "transcript": transcript, "stats": " ".join(out.split())[:200]}
    except Exception as exc:  # noqa: BLE001 — any surprise still has to become a fallback
        return {"success": False, "transcript": "", "error": f"mac backend error: {exc}"}
    finally:
        cleanup_remote()
        core.cleanup_prepared(prepared)


__all__ = ["transcribe", "probe", "settings_from", "DEFAULTS", "REMOTE_SCRIPT"]
