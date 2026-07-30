# audio-transcriber

Turn long audio recordings (1–3 hours and more — Android voice recorder files and the
like) into text using the OpenAI Audio API.

*Читать по-русски: [README.ru.md](README.ru.md)*

The API accepts files no larger than 25 MB, so every recording is automatically:

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

# 3. API key: copy .env.example to .env and paste your key
copy .env.example .env
```

The key can also come from an environment variable instead of `.env`:

```powershell
$env:OPENAI_API_KEY = "sk-proj-..."          # current session only
setx OPENAI_API_KEY "sk-proj-..."            # permanently (needs a new terminal)
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
>> [1/2] Recording_2026-07-28.m4a  (2:47:13, 96.4 MB, ~$1.00)
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
python transcribe.py --model gpt-transcribe      # most accurate text (no timecodes)
python transcribe.py --model gpt-4o-transcribe-diarize      # label who is speaking
python transcribe.py --prompt "Kubernetes, CI/CD, Grafana"  # hint for names and terms
```

### All options

| Option          | Default     | Purpose                                                           |
| --------------- | ----------- | ----------------------------------------------------------------- |
| `--model`       | `whisper-1` | which model recognises the speech — see the table below           |
| `--list-models` | —           | print what every model supports and exit                          |
| `--language`    | `ru`        | ISO-639-1 language code, or `auto`                                |
| `--prompt`      | —           | names, terms, spelling — noticeably improves accuracy             |
| `--speakers`    | —           | rename diarized speaker labels: `A=Alice,B=Client` (diarize only) |
| `--chunk`       | `900`       | target chunk length in seconds                                    |
| `--workers`     | `4`         | number of parallel requests (`1` — sequential, with context)      |
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

- **transient** (a real `429` rate limit, `500`, a dropped connection) — retried with a
  growing delay, honouring the `Retry-After` header;
- **unsupported parameter** (`400`) — the parameter is dropped and the request repeated;
- **hopeless** — out of credit (`insufficient_quota`), a bad key (`401`), no access to the
  model (`404`): retrying is pointless, so the script stops immediately, explains what to
  do, and does not push the remaining chunks through the same failure.
  Failed requests are not billed.

## Cost and time

- `whisper-1` — **$0.006 per minute** of audio: an hour ≈ $0.36, three hours ≈ $1.08.
  `gpt-transcribe` is cheaper — $0.0045/min, three hours ≈ $0.81.
  The script prints an estimate for the chosen model before it starts.
- Time: with `--workers 4`, three hours of audio take roughly 5–10 minutes.

## Choosing a model

Compare them straight from the script: `python transcribe.py --list-models`

| Model                               | Timecodes | Speakers | `--prompt` | $/min   |
| ----------------------------------- | --------- | -------- | ---------- | ------- |
| `whisper-1` (default)               | yes       | no       | yes        | $0.006  |
| `gpt-transcribe`                    | no        | no       | yes        | $0.0045 |
| `gpt-4o-transcribe`                 | no        | no       | yes        | $0.006  |
| `gpt-4o-mini-transcribe`            | no        | no       | yes        | $0.003  |
| `gpt-4o-mini-transcribe-2025-12-15` | no        | no       | yes        | $0.003  |
| `gpt-4o-transcribe-diarize`         | yes       | yes      | no         | $0.006  |

What that means in practice:

- **`whisper-1`** — Whisper V2. The only model that returns timecodes for each segment,
  which is why `.srt` and `.timestamped.txt` are produced with it only. This is the
  default.
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

- Want the **most accurate text** — `--model gpt-transcribe`. Cheaper than Whisper, too.
- Want **timecodes or subtitles** — `--model whisper-1` (the default). Whisper is less
  accurate on Russian than the GPT models, but it is the only one that gives you `.srt`.
- Want to know **who is speaking** — `--model gpt-4o-transcribe-diarize`.
- Want it **cheap at volume** — `--model gpt-4o-mini-transcribe`.

For Russian recordings `--prompt` helps especially: list the names, titles and terms that
appear in them, because without a hint the model spells them by ear and inconsistently
from one place to another.

## Diarization: who is speaking

Diarization splits a recording by voice: the model doesn't just turn speech into text, it
tags every line with a speaker label. For a meeting or a call you get a dialogue instead of
a solid wall of text:

```powershell
python transcribe.py --model gpt-4o-transcribe-diarize
```

By default speakers are labelled with letters:

```text
A: Afternoon — how's the release going?

B: In testing, shipping tomorrow. One regression left.

A: Good, keep me posted.
```

The letters can be swapped for names — after the transcription, based on who actually
turned out to be where:

```powershell
python transcribe.py --model gpt-4o-transcribe-diarize --speakers A=Alice,B=Client
```

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
- `--prompt` with the names, titles and acronyms from the recording tangibly reduces the
  number of mistakes in how they are spelled.
- `--workers 1` passes the tail of the previous chunk forward as context to the next one:
  slightly more coherent text across the seams, but processing is sequential and slower.
- If a recording is very quiet and the cut comes out "hard" (few pauses found in the log),
  lowering `--chunk` helps, e.g. `--chunk 600`.

## Requirements

- Python 3.9+
- FFmpeg and FFprobe on `PATH`
- `openai>=1.40`

## License

[MIT](LICENSE)
