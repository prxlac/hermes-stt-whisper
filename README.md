# whisper-asr

Self-hosted **whisper-asr-webservice** (`ahmetoner/whisper-asr-webservice`) as a Hermes
speech-to-text provider. Points at any `/asr` endpoint — default `http://localhost:9000`.

| | |
|---|---|
| Provider name | `whisper-asr` |
| Selection | `stt.provider: whisper-asr` |
| Config section | `stt.whisper-asr` |
| Credentials | none (endpoint URL only) |
| Dependencies | none (stdlib `urllib`) |

## Install

Via Hermes (git under the hood):

```bash
hermes plugins install prxlac/hermes-stt-whisper
hermes plugins enable whisper-asr
hermes config set --force stt.provider whisper-asr
```

Or clone manually — the plugin directory name is what `plugins.enabled` refers to, so keep it
`whisper-asr`:

```bash
git clone https://github.com/prxlac/hermes-stt-whisper.git ~/.hermes/plugins/whisper-asr
hermes plugins enable whisper-asr
hermes config set --force stt.provider whisper-asr
```

You still need a reachable whisper-asr-webservice endpoint; set `stt.whisper-asr.base_url` (below).

### Per-profile install

Profiles are isolated islands — `~/.hermes/plugins/`, `plugins.enabled` and `config.yaml`
are per-profile, and nothing is inherited from `default`. Two ways to see the plugin:

* **Config only** — the command shim points at this directory by absolute path, so a
  profile with just the `stt.*` config entries transcribes fine; the plugin itself stays
  invisible (`hermes plugins list`, and the plugin-registered fallback).
* **Full install** — symlink one source of truth, then enable + configure:

```bash
PY=/Users/alan/.hermes/hermes-agent/venv/bin/python3
PLUG=~/.hermes/plugins/whisper-asr
ln -s "$PLUG" ~/.hermes/profiles/<name>/plugins/whisper-asr
hermes --profile <name> plugins enable whisper-asr
for kv in stt.provider=whisper-asr stt.whisper-asr.base_url=http://localhost:9000 \
          stt.whisper-asr.language=hr stt.whisper-asr.timeout=300 \
          stt.providers.whisper-asr.type=command \
          "stt.providers.whisper-asr.command=$PY $PLUG/transcribe.py --base-url http://localhost:9000 {input_path} {output_path} {language} {model}"; do
  hermes --profile <name> config set --force "${kv%%=*}" "${kv#*=}"
done
```

The endpoint is passed inline (`--base-url`) on purpose: the command string is already
per-profile config, so the shim needs no config lookup and cannot read another profile's
file. For the Python plugin path, `load_config()` is profile-scoped.

## Config

```yaml
stt:
  provider: whisper-asr
  whisper-asr:
    base_url: http://localhost:9000   # or HERMES_WHISPER_ASR_URL
    language: hr                # "" -> server-side auto-detect
    task: transcribe            # transcribe | translate
    vad_filter: true
    word_timestamps: false
    timeout: 300
    initial_prompt: ""          # vocabulary hint (whisper's own prompt)
```

`language` is also resolved from `stt.language` / `stt.providers.whisper-asr.language` by
Hermes' dispatcher before reaching the provider; a 422 from the server on an explicit
language retries once with auto-detect.

## Desktop dropdown

The Voice tab's provider list is compiled into the desktop bundle plus any **command
provider** declared in config — a plugin-only install transcribes fine but is not
offered there. Declaring the shim below (same code, `transcribe.py`) puts `whisper-asr`
in the dropdown:

```bash
PY=/Users/alan/.hermes/hermes-agent/venv/bin/python3
PLUG=~/.hermes/plugins/whisper-asr
hermes config set --force stt.providers.whisper-asr.type command
hermes config set --force stt.providers.whisper-asr.command \
  "$PY $PLUG/transcribe.py --config ~/.hermes/config.yaml {input_path} {output_path} {language} {model}"
```

Resolution order at dispatch: built-in → command provider → plugin, so the shim wins
when declared and the plugin is the fallback.

The per-provider **model** row cannot be added to the Voice tab (the row list is
compiled TS); the server picks the model via its own `ASR_MODEL` env anyway.

## Verify

```bash
PY=/Users/alan/.hermes/hermes-agent/venv/bin/python3
say -o /tmp/probe.wav --data-format=LEI16@16000 "Pozdrav, test."
$PY ~/.hermes/plugins/whisper-asr/transcribe.py --config ~/.hermes/config.yaml /tmp/probe.wav /tmp/probe.txt hr
cat /tmp/probe.txt
```

## Notes

* `/asr` accepts 19+ container formats; `encode=true` (default) lets the server run the
  audio through ffmpeg, so `.caf`/`.webm` from the desktop recorder work.
* Client-direct voice is not used for plugin providers — the Desktop relays the audio to
  the gateway, which is where this provider runs.
