"""whisper-asr-webservice (ahmetoner/whisper-asr-webservice) client — stdlib only.

Deliberately imports nothing from Hermes: the command-provider shim
(``transcribe.py``) runs outside the repo's ``sys.path`` and shares this module.

Config lives under ``stt.whisper-asr`` in config.yaml::

    stt:
      provider: whisper-asr
      whisper-asr:
        base_url: http://localhost:9000
        language: hr
        vad_filter: true
        task: transcribe        # transcribe | translate
        timeout: 300

``base_url`` may also come from ``HERMES_WHISPER_ASR_URL``. A ``stt.providers.whisper-asr``
block is read as a fallback so a command-provider declaration and the plugin never
disagree about the endpoint.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_BASE_URL = "http://localhost:9000"
DEFAULT_TIMEOUT = 300.0
DEFAULT_TASK = "transcribe"
_ENV_BASE_URL = "HERMES_WHISPER_ASR_URL"
_SECTION = "whisper-asr"
# /asr output modes the endpoint accepts; we always ask for json so the text survives
# alongside the segments, and fall back to plain-body parsing for txt.
_JSON_OUTPUT = "json"


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _config_from_hermes() -> Dict[str, Any]:
    """The active profile's config dict.

    ``load_config`` is the loader the CLI/tools surfaces use and it handles profile
    scope and ``${VAR}`` expansion; a direct file read of ``$HERMES_HOME/config.yaml``
    is the fallback for the rare case where the import is unavailable.
    """
    try:
        from hermes_cli.config import load_config  # noqa: PLC0415 — only available inside Hermes

        config = load_config()
        if isinstance(config, dict):
            return config
    except Exception:  # noqa: BLE001 — never let config access break registration
        pass
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    path = Path(home).expanduser() / "config.yaml"
    try:
        if path.exists():
            import yaml  # noqa: PLC0415

            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Merge ``stt.<section>`` (canonical) and ``stt.providers.<section>`` from config.yaml.

    ``config_path`` (used by the command-provider shim, which runs outside Hermes) reads
    that file; without it the active profile's config comes from ``load_config()``.
    Never raises: an unreadable or malformed config yields the defaults so a broken
    YAML parse surfaces as a transcription error, not a plugin load error.
    """
    if config_path:
        path = Path(config_path).expanduser()
        try:
            if path.exists():
                import yaml  # available in the Hermes venv; absent when running standalone
                with path.open("r", encoding="utf-8") as handle:
                    raw = yaml.safe_load(handle) or {}
            else:
                raw = {}
        except Exception:  # noqa: BLE001 — config problems are reported by the caller path
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
    else:
        raw = _config_from_hermes()

    data: Dict[str, Any] = {}
    if isinstance(raw, dict):
        stt = raw.get("stt")
        stt = stt if isinstance(stt, dict) else {}
        providers = stt.get("providers")
        providers = providers if isinstance(providers, dict) else {}
        for block in (providers.get(_SECTION), stt.get(_SECTION)):
            if isinstance(block, dict):
                data.update(block)

    base_url = str(
        os.environ.get(_ENV_BASE_URL) or data.get("base_url") or DEFAULT_BASE_URL
    ).strip().rstrip("/")

    task = str(data.get("task") or DEFAULT_TASK).strip().lower()
    if task not in {"transcribe", "translate"}:
        task = DEFAULT_TASK

    order_raw = data.get("order")
    if isinstance(order_raw, str):
        order = [part.strip().lower() for part in order_raw.split(",") if part.strip()]
    elif isinstance(order_raw, (list, tuple)):
        order = [str(part).strip().lower() for part in order_raw if str(part).strip()]
    else:
        order = []

    return {
        "base_url": base_url,
        "task": task,
        "language": str(data.get("language") or "").strip(),
        "prompt": str(data.get("initial_prompt") or data.get("prompt") or "").strip(),
        "vad_filter": _truthy(data.get("vad_filter"), False),
        "word_timestamps": _truthy(data.get("word_timestamps"), False),
        "timeout": _number(data.get("timeout"), DEFAULT_TIMEOUT),
        "model": str(data.get("model") or "").strip(),
        "order": order,
        "mac": data.get("mac") if isinstance(data.get("mac"), dict) else {},
    }


def _multipart(field: str, file_path: Path) -> Tuple[bytes, str]:
    """Build a single-file multipart/form-data body without pulling in requests."""
    boundary = f"----hermeswhisper{uuid.uuid4().hex}"
    content_type = "application/octet-stream"
    if file_path.suffix.lower() in {".wav"}:
        content_type = "audio/wav"
    elif file_path.suffix.lower() in {".mp3"}:
        content_type = "audio/mpeg"
    elif file_path.suffix.lower() in {".m4a", ".mp4"}:
        content_type = "audio/mp4"
    elif file_path.suffix.lower() in {".ogg", ".opus"}:
        content_type = "audio/ogg"
    elif file_path.suffix.lower() in {".flac"}:
        content_type = "audio/flac"
    elif file_path.suffix.lower() in {".webm"}:
        content_type = "audio/webm"

    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{file_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + file_path.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


def prepare_wav16k(
    file_path: str,
    *,
    ffmpeg: str = "",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Convert any readable audio to a private 16 kHz mono PCM WAV.

    Both backends want that exact format: mlx needs it (the Mac has no ffmpeg) and the
    webservice fails *silently* on AAC/.m4a (HTTP 200 with an empty body), so converting
    once up front removes that trap for both.

    Returns ``{"path": ..., "workdir": ..., "error": None}`` or ``{"error": msg}``; the
    caller removes ``workdir``. Never raises.
    """
    import shutil  # noqa: PLC0415 — local: keeps module import cost at zero for the plugin
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    source = Path(file_path).expanduser()
    if not source.exists():
        return {"error": f"Audio file not found: {file_path}"}
    if source.stat().st_size == 0:
        return {"error": f"Audio file is empty: {file_path}"}

    binary = ffmpeg or shutil.which("ffmpeg") or ""
    if not binary:
        # The gateway may not inherit the user's PATH; Hermes ships its own ffmpeg.
        candidates = sorted(Path.home().glob(".hermes/tools/ffmpeg-*/ffmpeg"))
        binary = str(candidates[-1]) if candidates else ""
    if not binary:
        return {"error": "ffmpeg not found (set stt.whisper-asr.mac.ffmpeg)"}

    workdir = tempfile.mkdtemp(prefix="hermes-asr-")
    target = Path(workdir) / "in16k.wav"
    try:
        proc = subprocess.run(
            [binary, "-y", "-loglevel", "error", "-i", str(source), "-ac", "1",
             "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return {"error": f"ffmpeg conversion failed: {exc}"}
    if proc.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        detail = " ".join((proc.stderr or proc.stdout or "").split())[:300]
        shutil.rmtree(workdir, ignore_errors=True)
        return {"error": f"ffmpeg conversion failed: {detail or f'exit {proc.returncode}'}"}
    return {"path": str(target), "workdir": workdir, "error": None}


def cleanup_prepared(prepared: Dict[str, Any]) -> None:
    """Remove the temp dir handed out by :func:`prepare_wav16k`. Never raises."""
    import shutil  # noqa: PLC0415

    workdir = prepared.get("workdir") if isinstance(prepared, dict) else None
    if workdir:
        shutil.rmtree(str(workdir), ignore_errors=True)


def _post(url: str, body: bytes, content_type: str, timeout: float) -> Tuple[int, str]:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, detail


def _extract_text(body: str) -> str:
    """Transcript from a /asr body: JSON ``text`` when present, else the raw body."""
    stripped = body.strip()
    if not stripped:
        return ""
    if stripped[0] in "{[":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if error and not parsed.get("text"):
                raise RuntimeError(str(error))
            text = parsed.get("text")
            if isinstance(text, str):
                return text.strip()
            if isinstance(parsed.get("segments"), list):
                joined = " ".join(
                    str(seg.get("text", "")).strip()
                    for seg in parsed["segments"]
                    if isinstance(seg, dict)
                )
                return joined.strip()
        return stripped
    return stripped


def transcribe_file(
    file_path: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """POST *file_path* to ``{base_url}/asr``. Returns ``{"success", "transcript", "error"?}``.

    Never raises: every failure becomes an error string so both the plugin envelope
    and the command shim can report it without a traceback.
    """
    settings = settings or load_settings()
    path = Path(file_path).expanduser()
    if not path.exists():
        return {"success": False, "transcript": "", "error": f"Audio file not found: {file_path}"}
    if path.stat().st_size == 0:
        return {"success": False, "transcript": "", "error": f"Audio file is empty: {file_path}"}

    params: Dict[str, str] = {
        "task": settings.get("task") or DEFAULT_TASK,
        "output": _JSON_OUTPUT,
        "encode": "true",
    }
    lang = (language or settings.get("language") or "").strip()
    if lang:
        params["language"] = lang
    if settings.get("vad_filter"):
        params["vad_filter"] = "true"
    if settings.get("word_timestamps"):
        params["word_timestamps"] = "true"
    hint = (prompt or settings.get("prompt") or "").strip()
    if hint:
        params["initial_prompt"] = hint
    model_name = (model or settings.get("model") or "").strip()
    if model_name:
        params["model"] = model_name

    base_url = str(settings.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    timeout = _number(settings.get("timeout"), DEFAULT_TIMEOUT)
    body, content_type = _multipart("audio_file", path)

    def call(query: Dict[str, str]) -> Tuple[int, str]:
        url = f"{base_url}/asr?{urllib.parse.urlencode(query)}"
        return _post(url, body, content_type, timeout)

    try:
        status, response_body = call(params)
        # A 422 on an explicit language hint means the server's model rejects that code;
        # retry once with server-side auto-detect rather than failing the whole turn.
        if status == 422 and lang:
            retry = {k: v for k, v in params.items() if k != "language"}
            status, response_body = call(retry)
    except Exception as exc:  # noqa: BLE001 — urllib raises a zoo of connection errors
        return {
            "success": False,
            "transcript": "",
            "error": f"whisper-asr request to {base_url}/asr failed: {exc}",
        }

    if status != 200:
        detail = " ".join(response_body.split())[:400]
        return {
            "success": False,
            "transcript": "",
            "error": f"whisper-asr at {base_url}/asr returned HTTP {status}: {detail}",
        }
    try:
        text = _extract_text(response_body)
    except RuntimeError as exc:
        return {"success": False, "transcript": "", "error": f"whisper-asr: {exc}"}
    return {"success": True, "transcript": text}
