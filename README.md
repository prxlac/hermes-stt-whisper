# whisper-asr

Self-hosted **whisper-asr-webservice** (`ahmetoner/whisper-asr-webservice`) as a Hermes
speech-to-text provider, plus a **router** that prefers a Mac running mlx-whisper over
SSH (~28x realtime) and falls back to the webservice (~0.7x realtime) when the Mac is
asleep, off the network, or its venv is gone.

| | |
|---|---|
| Provider name | `whisper-asr` |
| Selection | `stt.provider: whisper-asr` |
| Config section | `stt.whisper-asr` |
| Credentials | none (endpoint URL only) |
| Dependencies | none (stdlib `urllib`/`subprocess`; `ssh`, `scp` and `ffmpeg` for the `mac` backend) |

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

## Routing: the Mac first, webservice as fallback

Each transcription is normalised to 16 kHz mono WAV (local ffmpeg) and then handed to the
backends listed in `stt.whisper-asr.order`; the first one that returns text wins.

```yaml
stt:
  whisper-asr:
    base_url: http://localhost:9000
    language: hr
    order: [mac, ws]                                  # backend ids: mac, ws
    mac:
      enabled: true
      host: mac-stt                                   # ssh alias for the Mac
      venv_python: ~/mlx-asr/venv/bin/python          # venv with mlx-whisper on it
      model: mlx-community/whisper-large-v3-turbo
      probe_timeout: 2                                # ssh ConnectTimeout for the probe
      down_ttl: 120                                   # remember a failed probe this long
      timeout: 300
      ffmpeg: ""                                      # "" -> PATH, then ~/.hermes/tools/ffmpeg-*
```

How it behaves:

* **Probe** — one `ssh -o BatchMode=yes <host> 'test -x <venv> && echo HERMES_ASR_OK'`.
  It checks reachability *and* that the venv still exists, so a broken Mac does not get a
  scp before failing. A failed probe is cached for `down_ttl`, so a sleeping Mac costs one
  short timeout instead of one per message.
* **Run** — scp the WAV plus `remote/mlx_transcribe.py` to `/tmp`, run that script with the
  mlx venv python, `cat` back `<base>.txt`, delete the remote files. The remote script
  loads the WAV with stdlib `wave` and passes a float32 array to
  `mlx_whisper.transcribe`, because the Mac has no ffmpeg and the CLI needs it.
* **Fallback** — any failure (probe, scp, non-zero mlx exit, empty transcript) moves to the
  next backend. Each attempt's error comes back in `attempts`, so a fallback is visible
  instead of looking like the fast path worked.
* **Forced backend** — `transcribe.py --backend ws` keeps a recording on the server;
  `--backend mac` skips the probe. `HERMES_WHISPER_ASR_DEBUG=1` prints
  `backend=... (mac=fail [reason], ws=ok)` on stderr.
* **Selftest** — `./selftest.sh` runs the four cases (forced mac, auto order, forced ws,
  fallback) against a synthesised clip.

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
