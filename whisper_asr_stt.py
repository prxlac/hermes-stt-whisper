"""TranscriptionProvider wrapper around the whisper-asr-webservice client.

The ABC lives in ``agent.transcription_provider``; the HTTP/config logic lives in
``whisper_asr_core`` so the command-provider shim can share it outside the repo's
``sys.path``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import whisper_asr_core as core

from agent.transcription_provider import TranscriptionProvider

PROVIDER_NAME = "whisper-asr"


class WhisperAsrProvider(TranscriptionProvider):
    """Self-hosted whisper-asr-webservice (faster-whisper / whisper backends)."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Whisper ASR (self-hosted)"

    def is_available(self) -> bool:
        """Config presence only — the ABC forbids network calls here.

        A dead endpoint surfaces as a transcription error the user can read; a
        probe on every picker paint would not.
        """
        settings = core.load_settings()
        return bool(settings.get("base_url"))

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "No API key — self-hosted endpoint",
            "env_vars": [
                {
                    "key": core._ENV_BASE_URL,
                    "prompt": "whisper-asr-webservice base URL",
                    "url": "",
                }
            ],
        }

    def transcribe(
        self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None, **extra: Any
    ) -> Dict[str, Any]:
        """Envelope: ``success``, ``transcript`` (empty on failure), ``provider``, ``error``."""
        prompt = extra.get("prompt")
        result = core.transcribe_file(
            file_path,
            settings=core.load_settings(),
            language=language,
            model=model,
            prompt=prompt if isinstance(prompt, str) else None,
        )
        envelope: Dict[str, Any] = {
            "success": bool(result.get("success")),
            "transcript": str(result.get("transcript") or ""),
            "provider": PROVIDER_NAME,
        }
        if not envelope["success"]:
            envelope["error"] = str(result.get("error") or "whisper-asr transcription failed")
        return envelope

    def list_models(self) -> List[Dict[str, Any]]:
        """The model is chosen server-side (ASR_MODEL) — there is nothing to enumerate."""
        return []


__all__ = ["WhisperAsrProvider", "PROVIDER_NAME"]
