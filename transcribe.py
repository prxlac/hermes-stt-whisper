#!/usr/bin/env python3
"""Command-provider shim: makes ``whisper-asr`` selectable in the Desktop Voice dropdown.

Hermes' Desktop Voice tab builds its provider list from a compiled bundle plus any
**command provider declared in config**, so a plugin-only install runs at runtime but
is not offered in that dropdown. Declaring (per profile, in that profile's config)::

    stt.providers.whisper-asr:
      type: command
      command: "<venv python> <plugin dir>/transcribe.py --base-url http://localhost:9000 \\
                {input_path} {output_path} {language} {model}"

routes through this shim, which uses the exact same client as the plugin
(``whisper_asr_core``). Resolution order is built-in → command provider → plugin, so
the shim wins when declared and the plugin is the fallback.

Settings come from ``--base-url`` (recommended: inlined in the command string, which is
already per-profile config) so the shim needs no config lookup at all. ``--config`` is an
optional fallback; without it the shim reads ``$HERMES_HOME/config.yaml``, then
``~/.hermes/config.yaml``.

Usage (placeholders are filled by Hermes)::

    transcribe.py [--base-url URL] [--config PATH] [--task T] [--vad] [--prompt P] \\
                  <input_audio> <output_path> [language] [model]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import whisper_asr_core as core  # noqa: E402


def _resolve_config_path(explicit: str) -> str:
    """Explicit path > ``$HERMES_HOME/config.yaml`` > ``~/.hermes/config.yaml`` ("" when none).

    The child env carries the served profile's home when one is bound, so the middle rung
    is the profile-correct one; the last is the single-profile CLI case, where Hermes may
    not export ``HERMES_HOME`` at all.
    """
    if explicit:
        return explicit
    candidates = []
    home = os.environ.get("HERMES_HOME")
    if home:
        candidates.append(Path(home).expanduser() / "config.yaml")
    candidates.append(Path.home() / ".hermes" / "config.yaml")
    for candidate in candidates:
        try:
            if candidate.exists():
                return str(candidate)
        except OSError:
            continue
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="whisper-asr STT command provider")
    parser.add_argument("--config", default="", help="config.yaml to read stt.whisper-asr from")
    parser.add_argument("--base-url", default="", help="override the endpoint (per-profile, inline)")
    parser.add_argument("--task", default="", help="transcribe | translate")
    parser.add_argument("--vad", action="store_true", help="enable server-side VAD filtering")
    parser.add_argument("--prompt", default="", help="initial_prompt vocabulary hint")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("language", nargs="?", default="")
    parser.add_argument("model", nargs="?", default="")
    args = parser.parse_args(argv)

    settings: Dict[str, Any] = core.load_settings(_resolve_config_path(args.config) or None)
    if args.base_url.strip():
        settings["base_url"] = args.base_url.strip().rstrip("/")
    if args.task.strip():
        settings["task"] = args.task.strip().lower()
    if args.vad:
        settings["vad_filter"] = True
    if args.prompt.strip():
        settings["prompt"] = args.prompt.strip()

    result = core.transcribe_file(
        args.input_path,
        settings=settings,
        language=args.language.strip() or None,
        model=args.model.strip() or None,
    )
    if not result.get("success"):
        print(f"whisper-asr: {result.get('error') or 'transcription failed'}", file=sys.stderr)
        return 1

    transcript = str(result.get("transcript") or "")
    Path(args.output_path).write_text(transcript, encoding="utf-8")
    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
