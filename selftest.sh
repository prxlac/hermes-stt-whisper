#!/bin/sh
# End-to-end check of the whisper-asr router: mac backend, auto order, ws forced, fallback.
#
#   ./selftest.sh        # all four cases
#   ./selftest.sh mac    # only the fast path (needs the Mac awake)
#
# Exit is non-zero when a required case fails. Each case prints which backend answered.
set -u

PY=${PY:-$HOME/.hermes/hermes-agent/venv/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
SHIM="$HERE/transcribe.py"
BASE_URL=${BASE_URL:-http://localhost:9000}
CONFIG=${CONFIG:-$HOME/.hermes/config.yaml}
ONLY=${1:-all}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/whisper-asr-selftest.XXXXXX")
TMP_CFG="$WORK/cfg-no-mac.yaml"
FAIL=0

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

say -o "$WORK/probe.aiff" "Ovo je test transkripcije preko Hermesa." 2>/dev/null \
  || say -o "$WORK/probe.aiff" "this is a transcription test" 2>/dev/null
ffmpeg -y -loglevel error -i "$WORK/probe.aiff" -ac 1 -ar 16000 -c:a pcm_s16le "$WORK/probe.wav" 2>/dev/null \
  || { echo "FAIL: ffmpeg missing"; exit 1; }

backend_of() {
  printf '%s' "$1" | sed -n 's/^whisper-asr: backend=\([a-z0-9]*\).*/\1/p' | head -1
}

run_case() {
  name=$1; expect=$2; shift 2
  printf '%-24s ' "$name"
  out=$(HERMES_WHISPER_ASR_DEBUG=1 "$PY" "$SHIM" --base-url "$BASE_URL" "$@" "$WORK/probe.wav" "$WORK/out.txt" hr 2>&1)
  got=$(backend_of "$out")
  if [ "$got" = "$expect" ]; then
    echo "ok (backend=$got)"
  else
    echo "FAIL (expected $expect, got '${got:-none}')"
    printf '%s\n' "$out" | tail -3
    FAIL=1
  fi
}

if [ "$ONLY" = "all" ] || [ "$ONLY" = "mac" ]; then
  run_case "forced mac" mac --backend mac
fi

if [ "$ONLY" = "all" ]; then
  run_case "auto order" mac
  run_case "forced ws" ws --backend ws

  # Same config with the mac backend switched off: must land on the webservice.
  "$PY" - "$CONFIG" "$TMP_CFG" <<'EOF'
import pathlib, sys, yaml
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
block = cfg.get("stt", {}).get("whisper-asr", {}) or {}
block["mac"] = dict(block.get("mac") or {}, enabled=False)
pathlib.Path(sys.argv[2]).write_text(yaml.safe_dump({"stt": {"whisper-asr": block}}))
EOF
  printf '%-24s ' "fallback (mac off)"
  out=$(HERMES_WHISPER_ASR_DEBUG=1 "$PY" "$SHIM" --config "$TMP_CFG" --base-url "$BASE_URL" \
        "$WORK/probe.wav" "$WORK/out.txt" hr 2>&1)
  got=$(backend_of "$out")
  if [ "$got" = "ws" ]; then echo "ok (backend=ws)"; else echo "FAIL (got '${got:-none}')"; FAIL=1; fi
fi

exit $FAIL
