# audio-transcriber

Turn long audio recordings (1–3 hours and more — Android voice recorder files and the
like) into text using speech-to-text models from [OpenRouter](https://openrouter.ai)
(MAI-Transcribe 2 by default) or the OpenAI Audio API.

*Читать по-русски: [README.ru.md](README.ru.md)*

The APIs accept files no larger than 25 MB, so every recording is automatically:

1. measured (`ffprobe`),
2. scanned for pauses (`ffmpeg silencedetect`),
3. cut into chunks **on those pauses**, so no word is split in half,
4. transcribed chunk by chunk — in parallel, with retries and an on-disk cache,
5. stitched back together into finished files.

## Install (once)

```powershell
# 1. FFmpeg (if you don't have it yet)
winget install Gyan.FFmpeg

# 2. Python dependencies
pip install -r requirements.txt

# 3. API keys: copy .env.example to .env and paste your keys
copy .env.example .env
```

Which key you need depends on the model:

- `OPENROUTER_API_KEY` — for the default model and every other OpenRouter model
  (get one at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys));
- `OPENAI_API_KEY` — only for the OpenAI models (`whisper-1`, `gpt-transcribe`,
  `gpt-4o-transcribe-diarize`, …).

The keys can also come from environment variables instead of `.env`:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."     # current session only
setx OPENROUTER_API_KEY "sk-or-v1-..."       # permanently (needs a new terminal)
```

## Usage

Drop your recordings into `input/` and run:

```powershell
python transcribe.py
```

Or double-click `run.bat` (it also accepts drag-and-dropped files).

You can see what is happening and how much is left — every stage has its own progress
bar with an ETA:

```text
>> [1/2] Recording_2026-07-28.m4a  (2:47:13, 96.4 MB, ~$0.28)
   scan pauses  [████████████████████████] 100% 812 pauses | took 0:41
   812 pauses found -> 12 chunks of ~0:13:56
   cut chunks   [██████████░░░░░░░░░░░░░░]  41% 5/12 | ETA 1:12
   transcribe   [██████████████░░░░░░░░░░]  58% 7/12 chunks | ETA 2:35
```

Results land in `output/` — three files per recording:

| File                   | What's inside                                          |
| ---------------------- | ------------------------------------------------------ |
| `name.txt`             | clean text, split into paragraphs on pauses            |
| `name.timestamped.txt` | the same text with `[0:02:00]` markers every 2 minutes |
| `name.srt`             | subtitles with exact timecodes                         |

### Common invocations

```powershell
python transcribe.py record.m4a                  # one file (a path, or a name from input/)
python transcribe.py D:\Recordings               # a whole folder
python transcribe.py --dry-run                   # show the cut plan, spend nothing
python transcribe.py --language en               # English
python transcribe.py --language auto             # detect the language
python transcribe.py --workers 8                 # faster (8 parallel requests)
python transcribe.py --workers 1                 # slower, but with context between chunks
python transcribe.py --force                     # overwrite an existing result
python transcribe.py --list-models               # what models exist and what they can do
python transcribe.py --model voxtral             # cleanest text (aliases: mai, voxtral, scribe)
python transcribe.py --model x-ai/grok-stt-2.0   # any OpenRouter speech-to-text model by id
python transcribe.py --model whisper-1           # OpenAI directly, the previous default
python transcribe.py --model gpt-4o-transcribe-diarize      # label who is speaking
python transcribe.py --model gpt-transcribe --prompt "Kubernetes, CI/CD"  # term hints (OpenAI only)
```

### All options

| Option          | Default     | Purpose                                                           |
| --------------- | ----------- | ----------------------------------------------------------------- |
| `--model`       | `mai`       | which model recognises the speech — see the table below           |
| `--list-models` | —           | print what every model supports and exit                          |
| `--language`    | `ru`        | ISO-639-1 language code, or `auto`                                |
| `--prompt`      | —           | names, terms, spelling (OpenAI models only — OpenRouter drops it) |
| `--diarize`     | —           | label who is speaking (with `mai`) — see Diarization below        |
| `--speakers`    | —           | rename speaker labels: `0=Alice,1=Client` (with `--diarize`)      |
| `--chunk`       | `900`       | target chunk length in seconds (`4792` with `--diarize`)          |
| `--workers`     | `4`         | parallel requests (`1` — sequential; OpenAI models carry context) |
| `--input-dir`   | `./input`   | where to look for recordings                                      |
| `--output-dir`  | `./output`  | where to write the text                                           |
| `--work-dir`    | `./work`    | temporary chunks and cache                                        |
| `--force`       | —           | re-transcribe even if a result already exists                     |
| `--keep-work`   | —           | keep chunks and cache after success                               |
| `--dry-run`     | —           | cut plan only, no API requests                                    |
| `--no-progress` | —           | no progress bars, plain log lines only                            |

## Resilience

Every chunk is saved to `work/<name>/chunks/` as soon as the API answers. If the process
dies, your connection drops or you hit Ctrl+C — just run the same command again: finished
chunks are read from the cache and you are not billed for them twice. After a successful
assembly `work/<name>/` is deleted (disable that with `--keep-work`).

The cache is bound to exactly what it holds: the file name carries the precise span
boundaries, the encoding parameters and a fingerprint of the request (model, language,
prompt). So when you change `--chunk` or the model, an old result cannot be substituted
for a new span — it simply isn't found, and leftover audio chunks from the previous cut
plan are deleted (recreating them is free). Chunks that still match are still reused.

Recordings that are already transcribed are skipped: if `output/name.txt` exists, the file
is not processed again without `--force`.

The script tells API errors apart by meaning:

- **transient** (a real `429` rate limit, `500`, a dropped connection, OpenRouter's `402`
  for too many requests in flight at once) — retried with a growing delay, honouring
  `Retry-After`;
- **unsupported parameter** (`400`) — the parameter is dropped and the request repeated;
- **hopeless** — out of credit (`insufficient_quota`, or OpenRouter's `402`), a bad key
  (`401`), no access to the model (`404`): retrying is pointless, so the script stops
  immediately, explains what to do, and does not push the remaining chunks through the
  same failure. Failed requests are not billed.

## Cost and time

- `mai` (the default) — **$0.10 per hour** of audio: three hours ≈ $0.30.
  `voxtral` — $0.24/hour, `scribe` — $0.22/hour, `whisper-1` — $0.36/hour.
  The script prints an estimate for the chosen model before it starts.
- Time: with `--workers 4`, three hours of audio take roughly 5–10 minutes.

## Choosing a model

Compare them straight from the script: `python transcribe.py --list-models`

| Model                                                | Via        | Timecodes | Speakers | `--prompt` | $/hour |
| ---------------------------------------------------- | ---------- | --------- | -------- | ---------- | ------ |
| `microsoft/mai-transcribe-2` (`mai`, default)        | OpenRouter | yes       | no       | no         | $0.10  |
| `mistralai/voxtral-small-transcribe-3-0` (`voxtral`) | OpenRouter | yes       | no       | no         | $0.24  |
| `elevenlabs/scribe-v2` (`scribe`)                    | OpenRouter | yes       | no       | no         | $0.22  |
| `whisper-1`                                          | OpenAI     | yes       | no       | yes        | $0.36  |
| `gpt-transcribe`                                     | OpenAI     | no        | no       | yes        | $0.27  |
| `gpt-4o-transcribe`                                  | OpenAI     | no        | no       | yes        | $0.36  |
| `gpt-4o-mini-transcribe`                             | OpenAI     | no        | no       | yes        | $0.18  |
| `gpt-4o-mini-transcribe-2025-12-15`                  | OpenAI     | no        | no       | yes        | $0.18  |
| `gpt-4o-transcribe-diarize`                          | OpenAI     | yes       | yes      | no         | $0.36  |

Any other OpenRouter speech-to-text model works too — pass its full id, e.g.
`--model x-ai/grok-stt-2.0` (see the
[collection](https://openrouter.ai/collections/speech-to-text-models)). The script checks
the id against OpenRouter's catalog before spending anything, suggests the closest match on
a typo, and asks for word timings so that `.srt` comes out whenever the model can do it.

### How the OpenRouter models were picked

They won a side-by-side test on real Russian recordings (a noisy two-person business
meeting and two clean consultations; ten models, the same fragments):

- **`mai`** — the most faithful: names and rare words come out right, the other person's
  short replies ("uh-huh", "right") become separate lines. Writes fillers ("um", "uh")
  verbatim. Took a 26-minute file in one request in 29 seconds.
- **`voxtral`** — the cleanest text: drops fillers and backchannel. Occasionally garbles
  overlapping speech or loses punctuation for a stretch.
- **`scribe`** — fluent and readable, but now and then swaps in a plausible wrong word
  ("five seas" instead of "five names", "don't close" instead of "don't open") — the
  hardest kind of error to spot. Timecodes are rebuilt from its per-word timings.
- `whisper-1` is fine on clean audio and clearly weaker on meetings; `gpt-transcribe` and
  Gemini dropped a "not" here and there (flipping the meaning) and give no timecodes;
  `openai/whisper-large-v3` invented whole phrases.

OpenRouter ignores `--prompt`, so with these models the hint is skipped with a warning.

### OpenAI models

- **`whisper-1`** — Whisper V2. Returns timecodes for each segment, so `.srt` and
  `.timestamped.txt` are produced. The default before the OpenRouter models arrived.
- **`gpt-transcribe`** — the newest and most accurate model for files, and cheaper than
  Whisper at that. Returns no timecodes: you get a single `.txt`. Pick it when you want the
  most accurate text and don't need subtitles.
- **`gpt-4o-transcribe`** — the previous generation, superseded by `gpt-transcribe`.
- **`gpt-4o-mini-transcribe`** — the cheapest, slightly less accurate. The dated version is
  a pinned snapshot (it won't shift under you when the model is updated).
- **`gpt-4o-transcribe-diarize`** — labels **who** said what (see below). Does not accept
  `--prompt` — an API limitation; the script warns and ignores it. Plus a hard limit of
  1400 seconds of audio per request, which the script takes into account when cutting.

Prices from the [OpenAI pricing page](https://developers.openai.com/api/docs/pricing),
checked 2026-07-30.

The script does not rely on that table rigidly: if the API reports that some parameter is
unsupported for the chosen model, the script drops it and repeats the request instead of
failing on every chunk. That matters, because OpenAI's model line-up changes faster than
SDK versions do.

### What to pick for Russian speech

Every model in the table supports Russian, and passing `--language ru` (the default)
noticeably helps accuracy and speed on all of them.

- Want the **most accurate text with subtitles** — the default, `mai`.
- Want **clean text to read**, without "um" and "uh-huh" — `--model voxtral`.
- Want to know **who is speaking** — `--diarize` (with the default `mai`).
- Need **term hints** via `--prompt` — an OpenAI model, e.g. `--model gpt-transcribe`.

## Diarization: who is speaking

Diarization splits a recording by voice: the model doesn't just turn speech into text, it
tags every line with a speaker label. For a meeting or a call you get a dialogue instead of
a solid wall of text:

```powershell
python transcribe.py --diarize
```

This uses MAI-Transcribe 2 (the default model) at its usual price, $0.10 per hour. The
whole recording goes to the model **in one request** while it fits (up to ~100 minutes), so
a speaker keeps the same label from the first minute to the last. A 40-minute consultation
took 20 seconds. Speakers are numbered from `0`:

```text
[0:00:02] 1: There are kids whose puberty is barely noticeable.
[0:00:08] 0: That's how it was with Raul, yes.
[0:00:11] 1: You can still talk things through with them.
```

The numbers say nothing about who is who — the model numbers voices in whatever order it
likes. Listen to (or read) the start, then name them:

```powershell
python transcribe.py --diarize --speakers 0=Mother,1=Consultant --force
```

A re-run pays again (pennies: $0.10 per hour) unless the first run had `--keep-work` — then
the cached answer is reused for free. In the dialogue the other person's short "uh-huh" and "right" become separate lines, which keeps
the turns honest but makes the text longer.

Recordings longer than ~100 minutes are cut into pieces, and then labels restart in every
piece — the script warns about that before sending anything.

### The OpenAI alternative

```powershell
python transcribe.py --model gpt-4o-transcribe-diarize --speakers A=Alice,B=Client
```

It labels speakers with letters, costs $0.36 per hour and is weaker on Russian.

**An important limitation.** The model assigns speaker labels independently in each chunk,
so `A` at the start of a recording and `A` an hour later are not necessarily the same
person. Because of that the script never merges lines across a chunk boundary, and prints a
warning when there is more than one chunk. Fewer, more consistent labels would mean bigger
chunks — but this model has two constraints.

First, a hard, undocumented limit of **1400 seconds (23:20) per request**: on a 44-minute
chunk the API answers `audio duration 2682.09 seconds is longer than 1400 seconds which is
the maximum for this model`. The script knows about it and won't let you exceed it — with
`--chunk 2700` the value is lowered automatically, with a warning.

Second, diarization answers noticeably slower than the other models: a 20-minute chunk
manages to hit the request timeout. So its chunk size is 15 minutes, audio is encoded more
compactly (32 kbit/s), and the timeout is raised to 20 minutes per request.

## Quality notes

- Android voice recorder files (`.m4a`, `.amr`, `.3gp`, `.opus`, `.wav`, …) are read
  directly, no manual conversion needed — every chunk is encoded to mono 16 kHz MP3.
- With OpenAI models, `--prompt` with the names, titles and acronyms from the recording
  tangibly reduces the number of mistakes in how they are spelled.
- With OpenAI models, `--workers 1` passes the tail of the previous chunk forward as
  context to the next one: slightly more coherent text across the seams, but processing is
  sequential and slower.
- If a recording is very quiet and the cut comes out "hard" (few pauses found in the log),
  lowering `--chunk` helps, e.g. `--chunk 600`.

## Requirements

- Python 3.9+
- FFmpeg and FFprobe on `PATH`
- `openai>=1.40`

## License

[MIT](LICENSE)
