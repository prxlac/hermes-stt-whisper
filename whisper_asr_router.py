"""Backend router: the Mac (mlx, ~28x realtime) first, the webservice as the fallback.

Both entry points (the Hermes plugin and the command-provider shim) go through
``transcribe_any`` so the two can never disagree about which machine did the work.

Order comes from ``stt.whisper-asr.order`` (list or comma string, default
``[mac, ws]``). A backend is attempted in turn; the first one that returns text wins.
Every attempt is recorded in ``attempts`` so a silent fallback is visible instead of
being mistaken for the fast path working.

Audio is normalised to 16 kHz mono WAV before either backend sees it: the webservice
answers HTTP 200 with an *empty body* on AAC/.m4a, and the Mac has no ffmpeg at all. An
empty transcript is treated as a failure while another backend is left to try.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import whisper_asr_core as core
import whisper_asr_mac as mac

DEFAULT_ORDER = ("mac", "ws")
KNOWN = {"mac", "ws"}


def resolve_order(settings: Dict[str, Any]) -> List[str]:
    raw = settings.get("order") or list(DEFAULT_ORDER)
    order = [name for name in raw if name in KNOWN]
    return order or list(DEFAULT_ORDER)


def _ws_transcribe(
    file_path: str,
    settings: Dict[str, Any],
    language: Optional[str],
    model: Optional[str],
    prompt: Optional[str],
) -> Dict[str, Any]:
    """Send a 16 kHz WAV to the webservice (converting first, see module docstring)."""
    ffmpeg = ""
    block = settings.get("mac")
    if isinstance(block, dict):
        ffmpeg = str(block.get("ffmpeg") or "")
    prepared = core.prepare_wav16k(file_path, ffmpeg=ffmpeg, timeout=min(settings.get("timeout", 300), 120))
    if prepared.get("error"):
        return {"success": False, "transcript": "", "error": prepared["error"]}
    try:
        return core.transcribe_file(
            prepared["path"],
            settings=settings,
            language=language,
            model=model,
            prompt=prompt,
        )
    finally:
        core.cleanup_prepared(prepared)


def transcribe_any(
    file_path: str,
    *,
    settings: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    backend: str = "auto",
) -> Dict[str, Any]:
    """Transcribe via the first backend that works.

    ``backend`` is ``auto`` (config order), ``mac`` or ``ws`` (forced, for tests and
    for "keep this on the server" requests). Returns
    ``{success, transcript, backend, attempts, error?}`` and never raises.
    """
    settings = settings or core.load_settings()
    forced = (backend or "auto").strip().lower()
    if forced in KNOWN:
        order = [forced]
    else:
        order = resolve_order(settings)

    attempts: List[Dict[str, Any]] = []
    empty_fallback: Optional[Dict[str, Any]] = None

    for index, name in enumerate(order):
        last = index == len(order) - 1
        if name == "mac":
            block = mac.settings_from(settings.get("mac"))
            if forced != "mac":
                available, detail = mac.probe(block)
                if not available:
                    attempts.append({"backend": "mac", "ok": False, "skipped": True, "error": detail})
                    continue
            result = mac.transcribe(
                file_path,
                settings=block,
                language=language or settings.get("language") or None,
                model=model,
            )
        else:
            result = _ws_transcribe(
                file_path,
                settings,
                language,
                model,
                prompt,
            )

        text = str(result.get("transcript") or "")
        attempts.append(
            {
                "backend": name,
                "ok": bool(result.get("success")) and bool(text.strip()),
                "error": result.get("error") or (None if text.strip() else "empty transcript"),
                "stats": result.get("stats"),
            }
        )
        if result.get("success") and text.strip():
            return {
                "success": True,
                "transcript": text,
                "backend": name,
                "attempts": attempts,
            }
        # Success but nothing said: keep the result, but give a remaining backend a shot
        # (an empty body is also how the webservice fails on formats it cannot decode).
        if result.get("success") and not text.strip() and not last:
            empty_fallback = empty_fallback or result

    if empty_fallback is not None:
        return {
            "success": True,
            "transcript": "",
            "backend": attempts[-1]["backend"],
            "attempts": attempts,
        }

    errors = "; ".join(
        f"{item['backend']}: {item.get('error') or 'failed'}" for item in attempts
    )
    return {
        "success": False,
        "transcript": "",
        "backend": None,
        "attempts": attempts,
        "error": errors or "no STT backend available",
    }


__all__ = ["transcribe_any", "resolve_order", "DEFAULT_ORDER", "KNOWN"]
