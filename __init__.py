"""Hermes plugin entry point: register whisper-asr as an STT provider.

Enabled via ``plugins.enabled`` (``hermes plugins enable whisper-asr``) and selected
with ``stt.provider: whisper-asr``. Built-in STT names always win over plugins, so
this name must not collide with ``local``/``local_command``/``groq``/``openai``/
``mistral``/``xai``/``elevenlabs``/``deepinfra``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("whisper-asr")

# Sibling modules are imported by plain name (the command-provider shim does the same),
# so this directory has to be importable before the first ``import whisper_asr_stt``.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from whisper_asr_stt import WhisperAsrProvider  # noqa: E402


def register(ctx) -> None:
    ctx.register_transcription_provider(WhisperAsrProvider())
    logger.debug("whisper-asr STT provider registered")
