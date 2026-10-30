"""Remote runner for the Mac (mlx-whisper) STT backend.

scp'd next to the audio by the ``whisper_asr_mac`` backend and executed with the mlx
venv python on that Mac. Reads a 16 kHz mono PCM WAV through stdlib ``wave`` and hands a
float32 numpy array straight to ``mlx_whisper.transcribe``, bypassing the CLI, which
needs ffmpeg and the Mac does not have it.

Usage: mlx_transcribe.py <in.wav> <out_base> [language|auto] [model]
Writes <out_base>.txt (one segment per line) and <out_base>.jsonl with timings.
"""

from __future__ import annotations

import json
import sys
import wave

import mlx_whisper
import numpy as np

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: mlx_transcribe.py <in.wav> <out_base> [language|auto] [model]", file=sys.stderr)
        return 2
    wav_path, out_base = argv[1], argv[2]
    language = argv[3] if len(argv) > 3 else "auto"
    model = argv[4] if len(argv) > 4 else DEFAULT_MODEL

    with wave.open(wav_path, "rb") as handle:
        if handle.getframerate() != 16000:
            print(f"expected 16 kHz WAV, got {handle.getframerate()}", file=sys.stderr)
            return 3
        if handle.getnchannels() != 1:
            print(f"expected mono WAV, got {handle.getnchannels()} channels", file=sys.stderr)
            return 3
        raw = handle.readframes(handle.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        print("empty audio", file=sys.stderr)
        return 4

    kwargs = {
        "path_or_hf_repo": model,
        "task": "transcribe",
        "verbose": False,
        "condition_on_previous_text": False,
    }
    if language and language != "auto":
        kwargs["language"] = language

    result = mlx_whisper.transcribe(audio, **kwargs)
    segments = result.get("segments") or []
    with open(out_base + ".jsonl", "w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(
                json.dumps(
                    {
                        "start": round(segment["start"], 2),
                        "end": round(segment["end"], 2),
                        "text": segment["text"].strip(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with open(out_base + ".txt", "w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(segment["text"].strip() + "\n")

    duration = len(audio) / 16000.0
    print(f"segments={len(segments)} audio_sec={duration:.0f} model={model} lang={language}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
