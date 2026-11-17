#!/usr/bin/env python3
"""
Transcribe long audio recordings (1-3+ hours, e.g. Android Recorder files)
into text using speech-to-text models from OpenRouter (default) or OpenAI.

Both APIs accept files up to 25 MB, so every recording is:
  1. probed for its duration (ffprobe),
  2. scanned for silence points (ffmpeg silencedetect),
  3. cut into chunks on those silence points (so no word is split in half),
  4. transcribed chunk-by-chunk (in parallel, with retries and an on-disk cache),
  5. stitched back together into .txt / .timestamped.txt / .srt outputs.

Usage:
    python transcribe.py                       # every new file in ./input
    python transcribe.py record.m4a            # one specific file
    python transcribe.py --list-models         # compare the available models
    python transcribe.py --model voxtral       # alias: mai | voxtral | scribe
    python transcribe.py --model x-ai/grok-stt-2.0   # any OpenRouter STT model
    python transcribe.py --model whisper-1     # OpenAI directly
    python transcribe.py --model gpt-4o-transcribe-diarize --speakers A=Denis,B=Client

Requires: ffmpeg + ffprobe on PATH, `pip install openai`, and an API key:
OPENROUTER_API_KEY for OpenRouter models, OPENAI_API_KEY for OpenAI ones
(env var or .env file next to this script).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    sys.exit("Package 'openai' is missing. Install it with:  pip install -r requirements.txt")


# --------------------------------------------------------------------------- #
# Configuration defaults
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = ROOT / "input"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_WORK_DIR = ROOT / "work"

# Audio extensions an Android recorder may produce (ffmpeg reads all of them).
AUDIO_EXTENSIONS = {
    ".m4a", ".mp3", ".mp4", ".aac", ".amr", ".3gp", ".3gpp", ".ogg", ".oga",
    ".opus", ".wav", ".wma", ".flac", ".webm", ".mkv", ".mpga", ".awb",
}

# Chunking. Target length keeps every chunk below the 25 MB API limit
# (16 kHz mono MP3 @ 48 kbps -> ~6 KB/s -> 15 min ~= 5.4 MB).
CHUNK_MIN_FACTOR = 0.6               # accept a silence cut from 60% of target
CHUNK_MAX_FACTOR = 1.25              # never let a chunk exceed 125% of target
MODEL_LIMIT_MARGIN = 10              # seconds of slack against a model's hard limit,
                                     # since an encoded chunk can run a frame long
SILENCE_NOISE_DB = -30               # silencedetect threshold
SILENCE_MIN_DURATION = 0.35          # seconds of quiet to count as a break

# Compact, API-friendly encoding for each chunk. Long chunks drop to a lower
# bitrate so that even a 60-minute piece stays under the upload limit.
CHUNK_SAMPLE_RATE = 16000
API_SIZE_LIMIT = 25 * 1024 * 1024


def bitrate_for(chunk_seconds: float, diarizing: bool = False) -> str:
    # Diarize requests are by far the slowest, so keep their uploads small.
    if diarizing:
        return "32k"
    return "48k" if chunk_seconds <= 1800 else "32k"


@dataclass(frozen=True)
class ModelSpec:
    """What one transcription model actually supports.

    Taken from the OpenAI audio API parameter reference: gpt-4o-transcribe and
    gpt-4o-mini-transcribe accept json only (no timestamps at all), whisper-1 is
    the only one with verbose_json/srt/vtt, and gpt-4o-transcribe-diarize needs
    diarized_json for speaker labels while rejecting prompt/temperature.
    """
    response_format: str
    timestamps: bool          # response carries per-segment start/end
    supports_prompt: bool
    supports_temperature: bool
    diarizes: bool = False
    needs_chunking_strategy: bool = False   # required for inputs over 30 s
    default_chunk_seconds: int = 900
    max_seconds: float | None = None        # hard per-request audio limit, if any
    price_per_minute: float | None = None   # rough estimate, see note below
    note: str = ""
    provider: str = "openai"                # "openai" (direct) or "openrouter"
    word_timestamps: bool = False           # segments must be rebuilt from word timings
    diarize_options: dict | None = None     # OpenRouter provider.options that turn on --diarize


# price_per_minute follows https://developers.openai.com/api/docs/pricing
# (checked 2026-07-30); it only drives the pre-run cost estimate.
MODEL_SPECS: dict[str, ModelSpec] = {
    "whisper-1": ModelSpec(
        response_format="verbose_json", timestamps=True,
        supports_prompt=True, supports_temperature=True,
        price_per_minute=0.006,
        note="Whisper V2; segment timestamps + .srt, accepts --prompt",
    ),
    "gpt-transcribe": ModelSpec(
        response_format="json", timestamps=False,
        supports_prompt=True, supports_temperature=False,
        price_per_minute=0.0045,
        note="newest and most accurate for files, cheapest of the GPT ones, no timestamps",
    ),
    "gpt-4o-transcribe": ModelSpec(
        response_format="json", timestamps=False,
        supports_prompt=True, supports_temperature=True,
        price_per_minute=0.006,
        note="previous generation, superseded by gpt-transcribe",
    ),
    "gpt-4o-mini-transcribe": ModelSpec(
        response_format="json", timestamps=False,
        supports_prompt=True, supports_temperature=True,
        price_per_minute=0.003,
        note="cheaper, slightly less accurate, no timestamps",
    ),
    "gpt-4o-mini-transcribe-2025-12-15": ModelSpec(
        response_format="json", timestamps=False,
        supports_prompt=True, supports_temperature=True,
        price_per_minute=0.003,
        note="pinned snapshot of gpt-4o-mini-transcribe",
    ),
    "gpt-4o-transcribe-diarize": ModelSpec(
        response_format="diarized_json", timestamps=True,
        supports_prompt=False, supports_temperature=False,
        diarizes=True, needs_chunking_strategy=True,
        # Two separate limits here. The API rejects anything over 1400 s
        # ("audio duration N seconds is longer than 1400 seconds which is the
        # maximum for this model" - undocumented but enforced), and these
        # requests are slow enough that a 20-minute chunk can hit the client
        # timeout. 900 s keeps both comfortable.
        default_chunk_seconds=900,
        max_seconds=1400,
        price_per_minute=0.006,
        note="labels who is speaking; ignores --prompt; max 1400 s per request",
    ),
    # OpenRouter models. Picked by a side-by-side test on Russian meetings and
    # consultations (2026-11-17); prices from the OpenRouter models API.
    # OpenRouter drops `prompt`, so none of them takes --prompt.
    "microsoft/mai-transcribe-2": ModelSpec(
        response_format="verbose_json", timestamps=True,
        supports_prompt=False, supports_temperature=True,
        price_per_minute=0.00167, provider="openrouter",
        diarize_options={"azure": {"diarization": {"enabled": True}}},
        note="best on Russian in our test; verbatim, splits the other speaker's turns",
    ),
    "mistralai/voxtral-small-transcribe-3-0": ModelSpec(
        response_format="verbose_json", timestamps=True,
        supports_prompt=False, supports_temperature=True,
        price_per_minute=0.004, provider="openrouter",
        note="cleanest text (drops um/uh), sometimes garbles crosstalk",
    ),
    "elevenlabs/scribe-v2": ModelSpec(
        response_format="verbose_json", timestamps=True,
        supports_prompt=False, supports_temperature=True,
        price_per_minute=0.00367, provider="openrouter", word_timestamps=True,
        note="fluent, but may swap in plausible wrong words",
    ),
}

# Short names for the models that won the comparison.
MODEL_ALIASES: dict[str, str] = {
    "mai": "microsoft/mai-transcribe-2",
    "voxtral": "mistralai/voxtral-small-transcribe-3-0",
    "scribe": "elevenlabs/scribe-v2",
}

DEFAULT_MODEL = "microsoft/mai-transcribe-2"

# --diarize: the whole recording in one request while it fits, so speaker
# labels stay consistent. 6000 s at 32 kbit/s is ~24 MB, under the 25 MB cap.
DIARIZE_CHUNK_SECONDS = 4800
DIARIZE_MAX_SECONDS = 6000


def diarize_extra_body(options: dict) -> dict:
    """Request fields that switch on a provider's diarization through OpenRouter.

    The SDK sends audio as multipart, which silently drops a nested object, so
    `provider` goes as a JSON string - OpenRouter parses it (checked 2026-11-17).
    """
    return {"provider": json.dumps({"options": options})}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CATALOG_URL = f"{OPENROUTER_BASE_URL}/models?output_modalities=transcription"

# Where each provider lives and which key it needs. Both speak the OpenAI
# audio API, so one SDK client class serves both.
PROVIDERS: dict[str, dict] = {
    "openai": {
        "name": "OpenAI",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "credits_url": "https://platform.openai.com/settings/organization/billing",
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": OPENROUTER_BASE_URL,
        "key_env": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/settings/keys",
        "credits_url": "https://openrouter.ai/settings/credits",
    },
}


def catalog_prices(data: dict) -> dict[str, float | None]:
    """Model id -> $ per minute from an OpenRouter models listing.

    Duration-priced models carry a per-second `prompt` price; token-priced ones
    also carry `completion`, and their per-minute cost cannot be known upfront.
    """
    prices: dict[str, float | None] = {}
    for model in data.get("data") or []:
        pricing = model.get("pricing") or {}
        try:
            per_second = float(pricing.get("prompt") or 0)
            per_token_out = float(pricing.get("completion") or 0)
        except (TypeError, ValueError):
            per_second, per_token_out = 0.0, 0.0
        prices[model["id"]] = per_second * 60 if per_second and not per_token_out else None
    return prices


def fetch_openrouter_catalog(timeout: float = 20.0) -> dict[str, float | None]:
    """Every speech-to-text model OpenRouter offers. Raises OSError when offline."""
    import urllib.request
    with urllib.request.urlopen(OPENROUTER_CATALOG_URL, timeout=timeout) as response:
        return catalog_prices(json.load(response))


def resolve_model(name: str, catalog_loader=fetch_openrouter_catalog) -> str:
    """Turn an alias or model id into a full id that is present in MODEL_SPECS.

    A `vendor/model` id that is not built in is checked against OpenRouter's
    catalog and registered with a generic spec, so any OpenRouter model works.
    """
    model = MODEL_ALIASES.get(name.strip().lower(), name.strip())
    if model in MODEL_SPECS:
        return model
    if "/" not in model:
        close = difflib.get_close_matches(model, [*MODEL_ALIASES, *MODEL_SPECS], n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(f"unknown model '{name}'.{hint} Run --list-models to see the choices, "
                         "or pass a full OpenRouter id such as x-ai/grok-stt-2.0")

    price: float | None = None
    try:
        catalog = catalog_loader()
    except Exception as error:  # noqa: BLE001 - an optional check must never stop the run
        log(f"warning: could not check '{model}' against the OpenRouter catalog ({error!r}); "
            "trying it anyway")
    else:
        if model not in catalog:
            close = difflib.get_close_matches(model, list(catalog), n=3, cutoff=0.6)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"OpenRouter has no speech-to-text model '{model}'.{hint} "
                             "See https://openrouter.ai/collections/speech-to-text-models")
        price = catalog[model]

    MODEL_SPECS[model] = ModelSpec(
        response_format="verbose_json", timestamps=True,
        supports_prompt=False, supports_temperature=True,
        price_per_minute=price, provider="openrouter", word_timestamps=True,
        note="any OpenRouter model; falls back to plain text if it has no timestamps",
    )
    return model

MAX_ATTEMPTS = 4                     # per chunk, on top of the SDK's own retries
MAX_INFLIGHT_WAITS = 10              # OpenRouter 402 "too many in flight" waits per chunk
MAX_ADAPTATIONS = 3                  # optional params dropped after an API 400
TICK_SECONDS = 1.0                   # progress-bar refresh while waiting on the API
HEARTBEAT_SECONDS = 30               # keep-alive line when output is redirected
REQUEST_TIMEOUT = 1200.0             # per request; diarization can be slow to answer
PROMPT_CONTEXT_CHARS = 700           # tail of previous text fed forward (sequential mode)


# --------------------------------------------------------------------------- #
# Console output: log lines + one live progress bar
# --------------------------------------------------------------------------- #

_ACTIVE_BAR: Progress | None = None


def log(message: str) -> None:
    """Print a line without leaving a half-drawn progress bar behind it."""
    bar = _ACTIVE_BAR
    if bar is not None:
        bar.clear_line()
    print(message, flush=True)
    if bar is not None:
        bar.redraw()


class Progress:
    """A single-line progress bar with ETA, safe to advance from worker threads.

    Used as a context manager. On a real terminal it redraws in place; when
    output is redirected it falls back to occasional one-line updates.
    """

    def __init__(self, total: float, label: str, enabled: bool = True, width: int = 24):
        self.total = max(1e-9, float(total))
        self.label = label
        self.width = width
        self.value = 0.0
        self.note = ""
        self.started = time.monotonic()
        self.enabled = enabled
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._lock = threading.RLock()
        self._last_len = 0
        self._last_draw = 0.0
        self._last_percent = -1
        self._closed = False
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._full, self._empty = ("█", "░") if self._supports_blocks() else ("#", "-")

    @staticmethod
    def _supports_blocks() -> bool:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            "█░".encode(encoding)
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    # -- lifecycle ---------------------------------------------------------- #

    def __enter__(self) -> Progress:
        global _ACTIVE_BAR
        if self.enabled:
            _ACTIVE_BAR = self
            self._draw(force=True)
            # A long API call can leave the bar untouched for minutes; keep the
            # elapsed clock moving so a slow request never looks like a freeze.
            self._ticker = threading.Thread(target=self._tick, daemon=True)
            self._ticker.start()
        return self

    def _tick(self) -> None:
        while not self._stop.wait(TICK_SECONDS):
            self._draw(force=True)

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close(completed=exc_type is None)
        return False

    def close(self, completed: bool = True) -> None:
        global _ACTIVE_BAR
        self._stop.set()
        with self._lock:
            if self._closed:
                return
            if completed:
                self.value = self.total
            self._closed = True
            if self.enabled:
                self._draw(force=True, closing=True)
                sys.stdout.write("\n")
                sys.stdout.flush()
            if _ACTIVE_BAR is self:
                _ACTIVE_BAR = None

    # -- updates ------------------------------------------------------------ #

    def advance(self, amount: float = 1.0, note: str | None = None) -> None:
        with self._lock:
            self.value = min(self.total, self.value + amount)
            if note is not None:
                self.note = note
        self._draw()

    def set_value(self, value: float, note: str | None = None) -> None:
        with self._lock:
            self.value = max(0.0, min(self.total, float(value)))
            if note is not None:
                self.note = note
        self._draw()

    def set_note(self, note: str) -> None:
        with self._lock:
            self.note = note
        self._draw(force=True)

    # -- drawing ------------------------------------------------------------ #

    def clear_line(self) -> None:
        with self._lock:
            if self.enabled and self.tty and self._last_len:
                sys.stdout.write("\r" + " " * self._last_len + "\r")
                sys.stdout.flush()
                self._last_len = 0

    def redraw(self) -> None:
        if not self._closed:
            self._draw(force=True)

    def _render(self, closing: bool = False) -> str:
        fraction = min(1.0, self.value / self.total)
        filled = int(round(fraction * self.width))
        bar = self._full * filled + self._empty * (self.width - filled)
        elapsed = time.monotonic() - self.started
        if closing or fraction >= 0.999:
            tail = f"took {fmt_short(elapsed)}"
        elif fraction > 0.02:
            tail = f"{fmt_short(elapsed)} elapsed, ETA {fmt_short(elapsed * (1 - fraction) / fraction)}"
        else:
            tail = f"{fmt_short(elapsed)} elapsed, ETA --:--"
        note = f" {self.note}" if self.note else ""
        return f"   {self.label} [{bar}] {fraction * 100:3.0f}%{note} | {tail}"

    def _draw(self, force: bool = False, closing: bool = False) -> None:
        with self._lock:
            if not self.enabled or (self._closed and not closing):
                return
            now = time.monotonic()
            percent = int(min(1.0, self.value / self.total) * 100)
            if self.tty:
                if not force and now - self._last_draw < 0.2 and percent == self._last_percent:
                    return
                line = self._render(closing)
                sys.stdout.write("\r" + line + " " * max(0, self._last_len - len(line)))
                sys.stdout.flush()
                self._last_len = len(line)
            else:
                # Redirected output: one line per 10% step, no carriage returns.
                # 100% is left to close(), so the final line appears exactly once.
                # A heartbeat line still goes out if nothing moved for a while,
                # so a piped log also shows the run is alive.
                stale = now - self._last_draw >= HEARTBEAT_SECONDS
                if not closing and not stale and (percent >= 100 or percent < self._last_percent + 10):
                    return
                print(self._render(closing), flush=True)
            self._last_draw = now
            self._last_percent = percent


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def fmt_short(seconds: float) -> str:
    """Compact clock for elapsed/ETA: 4:07 under an hour, 1:12:30 above it."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def fmt_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._")
    return slug or "recording"


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, no dependency on python-dotenv."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(
            f"'{name}' was not found on PATH.\n"
            "Install FFmpeg and reopen the terminal, e.g.:  winget install Gyan.FFmpeg"
        )
    return path


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


# --------------------------------------------------------------------------- #
# Audio inspection and chunking
# --------------------------------------------------------------------------- #

def probe_duration(path: Path) -> float:
    result = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"ffprobe could not read the duration of {path.name}:\n{result.stderr.strip()}")


TIME_RE = re.compile(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)")
SILENCE_RE = re.compile(r"silence_(start|end):\s*(-?[\d.]+)")


def detect_silences(path: Path, duration: float, show_progress: bool = True) -> list[tuple[float, float]]:
    """Return [(silence_start, silence_end), ...], reporting scan progress live."""
    process = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
            "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_DURATION}",
            "-f", "null", "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    starts: list[float] = []
    silences: list[tuple[float, float]] = []
    tail: list[str] = []

    # text mode turns ffmpeg's \r status updates into ordinary lines
    with Progress(duration, "scan pauses ", enabled=show_progress) as bar:
        for line in process.stderr:
            tail.append(line)
            del tail[:-40]
            for kind, value in SILENCE_RE.findall(line):
                if kind == "start":
                    starts.append(float(value))
                elif starts:
                    silences.append((starts.pop(), float(value)))
            moment = TIME_RE.search(line)
            if moment:
                h, m, s = moment.groups()
                bar.set_value(int(h) * 3600 + int(m) * 60 + float(s),
                              note=f"{len(silences)} pauses")
        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg could not scan {path.name}:\n{''.join(tail).strip()[-2000:]}")
    return silences


def plan_cut_points(duration: float, silences: list[tuple[float, float]],
                    target: float, hard_max: float | None = None) -> list[tuple[float, float]]:
    """Split [0, duration] into (start, end) spans, preferring silent boundaries.

    `hard_max` is a per-request limit the model enforces; no span may exceed it.
    """
    min_len, max_len = target * CHUNK_MIN_FACTOR, target * CHUNK_MAX_FACTOR
    if hard_max:
        max_len = min(max_len, hard_max)
        min_len = min(min_len, max_len * 0.75)
    # Cut in the middle of each silent stretch.
    candidates = sorted((start + end) / 2 for start, end in silences)

    spans: list[tuple[float, float]] = []
    position = 0.0
    while duration - position > max_len:
        window = [c for c in candidates if position + min_len <= c <= position + max_len]
        cut = min(window, key=lambda c: abs(c - (position + target))) if window else position + target
        spans.append((position, cut))
        position = cut
    spans.append((position, duration))
    return spans


def extract_chunk(source: Path, destination: Path, start: float, end: float,
                  bitrate: str = "48k") -> None:
    """Decode [start, end) of `source` into a compact mono MP3."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = run_command([
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{max(0.05, end - start):.3f}",
        "-i", str(source),
        "-vn", "-ac", "1", "-ar", str(CHUNK_SAMPLE_RATE),
        "-c:a", "libmp3lame", "-b:a", bitrate,
        str(destination),
    ])
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"ffmpeg failed to extract {start:.1f}-{end:.1f}s:\n{result.stderr.strip()[-2000:]}")
    if destination.stat().st_size > API_SIZE_LIMIT:
        raise RuntimeError(
            f"Chunk {destination.name} is {destination.stat().st_size / 1e6:.1f} MB, over the 25 MB API limit. "
            "Re-run with a smaller --chunk value."
        )


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    index: int
    start: float
    end: float
    audio_path: Path
    cache_path: Path

    @property
    def label(self) -> str:
        return f"[{self.index + 1:03d}] {fmt_duration(self.start)}-{fmt_duration(self.end)}"


def request_signature(model: str, language: str | None, prompt: str | None,
                      diarize: bool = False) -> str:
    """Identity of everything that changes what a transcription result means."""
    parts = [model, language or "auto", prompt or ""]
    if diarize:
        parts.append("diarize")   # appended only when on, so existing caches keep their names
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def chunk_paths(chunk_dir: Path, index: int, start: float, end: float,
                signature: str, bitrate: str) -> tuple[Path, Path]:
    """File names that encode the exact time span they hold.

    Cached work is only ever reused for the identical span, encoding and request
    parameters. Change --chunk (or the model, language, prompt) and the names
    change too, so a stale chunk can never be sent under a new span's label.
    """
    stamp = f"{int(round(start * 1000)):09d}-{int(round(end * 1000)):09d}"
    audio = chunk_dir / f"{index:03d}_{stamp}_{CHUNK_SAMPLE_RATE}_{bitrate}.mp3"
    cache = chunk_dir / f"{index:03d}_{stamp}_{signature}.json"
    return audio, cache


def foreign_cache_count(chunk_dir: Path, signature: str) -> int:
    """Cached results in `chunk_dir` made with other request settings (model, language...)."""
    if not chunk_dir.is_dir():
        return 0
    return sum(1 for path in chunk_dir.glob("*.json") if not path.stem.endswith(f"_{signature}"))


def prune_stale_audio(chunk_dir: Path, keep: set[str]) -> int:
    """Delete chunk audio left over from a different cut plan (it is free to redo)."""
    removed = 0
    for leftover in chunk_dir.glob("*.mp3"):
        if leftover.name not in keep:
            leftover.unlink(missing_ok=True)
            removed += 1
    return removed


class FatalAPIError(RuntimeError):
    """An API failure no retry can fix: no credit, bad key, no model access."""


# Set as soon as one chunk hits a fatal error, so the other workers stop
# immediately instead of burning through the same failure in parallel.
_ABORT = threading.Event()


def _pause(seconds: float) -> None:
    """Sleep before a retry, but wake at once when another chunk hit a fatal error."""
    _ABORT.wait(seconds)

FATAL_HINTS = {
    "insufficient_quota":
        "the OpenAI account is out of credit (insufficient_quota). Failed requests are "
        "not billed. Add funds at https://platform.openai.com/settings/organization/billing "
        "then re-run - the chunks already cut are reused.",
    "account_deactivated":
        "this OpenAI account is deactivated.",
    "model_not_found":
        "the account cannot access this model. Run --list-models and pick another one.",
}


def error_metadata(error: Exception) -> dict:
    """OpenRouter's `metadata` object from an error body, or {}."""
    body = getattr(error, "body", None)
    metadata = body.get("metadata") if isinstance(body, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def is_in_flight_wait(error: Exception) -> bool:
    """OpenRouter's 402 for too many concurrent requests on a small balance."""
    return (getattr(error, "status_code", None) == 402
            and error_metadata(error).get("reason") == "in_flight_budget_exhausted")


def fatal_reason(error: Exception, model: str) -> str | None:
    """Explain why retrying is pointless, or None if the error is worth a retry."""
    spec = MODEL_SPECS.get(model)
    provider = PROVIDERS[spec.provider if spec else "openai"]
    # OpenRouter reports numeric codes (402), OpenAI string ones (insufficient_quota).
    code = str(getattr(error, "code", None) or "").strip()
    status = getattr(error, "status_code", None)
    if status == 402:
        # Too many requests in flight for the balance: they settle, then it works.
        if is_in_flight_wait(error):
            return None
        return (f"your {provider['name']} balance or this key's spending limit is used up "
                f"({api_message(error)}). Add funds at {provider['credits_url']} or raise the "
                "key limit, then re-run - the chunks already cut are reused.")
    if code in FATAL_HINTS:
        return FATAL_HINTS[code]
    if status == 401 or code == "invalid_api_key":
        return f"{provider['key_env']} was rejected. Check the key at {provider['key_url']}"
    if status == 403:
        return (f"{provider['name']} refused the request (403): {api_message(error)}. "
                "This can be the account, the region or content moderation.")
    if status == 404:
        return (f"model '{model}' does not exist or is not enabled for this account. "
                "Run --list-models and pick another one.")
    return None


def retry_after_seconds(error: Exception) -> float | None:
    """Honour the server's Retry-After header when it sends one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers else None
    if not value:
        # OpenRouter also repeats it inside the error body.
        value = (error_metadata(error).get("headers") or {}).get("Retry-After")
    try:
        return min(120.0, float(value)) if value else None
    except (TypeError, ValueError):
        return None


# Optional request parameters, ordered most specific first, that may be dropped
# if the API reports them as unsupported for the chosen model.
DROPPABLE_OPTIONS = ("timestamp_granularities", "chunking_strategy", "temperature", "prompt", "language")


PROTECTED_OPTIONS = {"model", "file", "extra_body"}   # SDK plumbing, never an API complaint


DURATION_LIMIT_RE = re.compile(r"longer than ([\d.]+) seconds")


def duration_limit_from_error(error: Exception) -> float | None:
    """Pull the model's own limit out of e.g. '... is longer than 1400 seconds ...'."""
    match = DURATION_LIMIT_RE.search(api_message(error))
    return float(match.group(1)) if match else None


def api_message(error: Exception) -> str:
    """The human-readable message the API returned, without the JSON wrapper."""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("message"):
        message = str(body["message"]).strip()
        # OpenRouter wraps upstream failures as "Provider returned error" and
        # keeps the provider's own words in metadata.raw - that is the useful part.
        raw = error_metadata(error).get("raw")
        if raw:
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            message = f"{message}: {raw_text.strip()[:500]}"
        return message
    return str(getattr(error, "message", None) or error).strip()


def drop_unsupported_option(options: dict, error: Exception,
                            keep: set[str] = frozenset()) -> str | None:
    """Remove whichever optional parameter the API complained about.

    Returns the removed key, or None if the error is about something else.
    """
    # Most reliable: the API names the offending parameter in error.param.
    blocked = PROTECTED_OPTIONS | set(keep)
    param = str(getattr(error, "param", None) or "").strip()
    if param in options and param not in blocked:
        options.pop(param)
        return param

    message = api_message(error).lower()
    # Next best: a parameter name in quotes, e.g. "Unknown parameter: 'language'".
    for quoted in re.findall(r"['\"]([a-z_]+)['\"]", message):
        if quoted in options and quoted not in blocked:
            options.pop(quoted)
            return quoted

    for key in DROPPABLE_OPTIONS:
        if key in options and key in message:
            options.pop(key)
            return key
    if ("response_format" in message and options.get("response_format") != "json"
            and "response_format" not in keep):
        options["response_format"] = "json"   # last resort: plain text, no timestamps
        return "response_format"
    return None


def read_cache(chunk: Chunk) -> dict | None:
    """Previously transcribed chunk, or None if absent, corrupt or for another span."""
    if not chunk.cache_path.is_file():
        return None
    try:
        cached = json.loads(chunk.cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("text") is None:
        return None
    # Belt and braces: the file name already encodes the span, but never trust a
    # result whose recorded boundaries disagree with the chunk being asked for.
    if (abs(float(cached.get("start", -1)) - chunk.start) > 0.5
            or abs(float(cached.get("end", -1)) - chunk.end) > 0.5):
        return None
    return cached


WORD_PAUSE_SPLIT = 0.8         # seconds of silence between words that start a new segment
WORD_SEGMENT_SOFT_MAX = 15.0   # past this length, split after the next sentence end
WORD_SEGMENT_HARD_MAX = 30.0   # never let a segment grow longer than this
SENTENCE_END = (".", "?", "!", "…")


def has_timing(word: dict) -> bool:
    """True when a word carries numeric start/end (some providers leave them null)."""
    try:
        float(word["start"]), float(word["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def segments_from_words(words: list[dict], offset: float) -> list[dict]:
    """Group word timings into subtitle-sized segments, shifted by `offset` seconds.

    For models (Scribe) that time every word but return the whole chunk as one
    segment: a new segment starts on a pause, after a sentence once the current
    one is long, and in any case before it would exceed the hard maximum.
    """
    segments: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            segments.append({
                "start": float(current[0]["start"]) + offset,
                "end": float(current[-1]["end"]) + offset,
                "text": " ".join(word["word"].strip() for word in current),
            })
            current.clear()

    for word in words:
        if not (word.get("word") or "").strip() or not has_timing(word):
            continue
        if current:
            start = float(current[0]["start"])
            pause = float(word["start"]) - float(current[-1]["end"])
            long_enough = float(current[-1]["end"]) - start >= WORD_SEGMENT_SOFT_MAX
            sentence_over = current[-1]["word"].strip().endswith(SENTENCE_END)
            too_long = float(word["end"]) - start > WORD_SEGMENT_HARD_MAX
            if pause >= WORD_PAUSE_SPLIT or (long_enough and sentence_over) or too_long:
                flush()
        current.append(word)
    flush()
    return segments


def payload_segments(payload: dict, offset: float, end: float | None = None) -> list[dict]:
    """Absolute-time segments from an API response, rebuilt from words if needed.

    A response with text but no timings at all becomes one segment spanning
    the chunk [offset, end], so its text is never dropped from the output.
    """
    raw_segments = payload.get("segments") or []
    words = payload.get("words") or []
    has_speakers = any(segment.get("speaker") is not None for segment in raw_segments)
    if words and len(raw_segments) <= 1 and not has_speakers:
        from_words = segments_from_words(words, offset)
        if from_words:
            return from_words

    segments = []
    for segment in raw_segments:
        entry = {
            "start": float(segment.get("start", 0.0)) + offset,
            "end": float(segment.get("end", 0.0)) + offset,
            "text": (segment.get("text") or "").strip(),
        }
        if segment.get("speaker") is not None:   # MAI numbers speakers from 0
            entry["speaker"] = str(segment["speaker"])
        segments.append(entry)

    text = (payload.get("text") or "").strip()
    if not segments and text:
        segments.append({"start": float(offset),
                         "end": float(end if end is not None else offset), "text": text})
    return segments


def transcribe_chunk(client: OpenAI, chunk: Chunk, model: str, language: str | None,
                     prompt: str | None, diarize: bool = False, concurrent: bool = True) -> dict:
    """Send one chunk to the API and return {'text': str, 'segments': [...]}.

    `diarize` asks the provider for speaker labels; `concurrent` says whether other
    chunks are in flight at the same time (it only shapes the advice in errors).
    """
    cached = read_cache(chunk)
    if cached is not None:
        return cached

    spec = MODEL_SPECS[model]
    options: dict = {"model": model, "response_format": spec.response_format}
    if language:
        options["language"] = language
    if prompt and spec.supports_prompt:
        options["prompt"] = prompt[-1000:]
    if spec.supports_temperature:
        options["temperature"] = 0
    if spec.needs_chunking_strategy:
        # Required by gpt-4o-transcribe-diarize for anything over 30 seconds:
        # the server normalizes loudness and picks boundaries by VAD itself.
        options["chunking_strategy"] = "auto"
    if diarize and spec.diarize_options:
        options["extra_body"] = diarize_extra_body(spec.diarize_options)
    if spec.word_timestamps:
        # Scribe returns one segment per request; words let us rebuild real ones.
        options["timestamp_granularities"] = ["word", "segment"]

    attempt = 0
    adaptations = 0
    inflight_waits = 0
    while True:
        if _ABORT.is_set():
            raise FatalAPIError("stopped after a fatal API error on another chunk")
        try:
            with chunk.audio_path.open("rb") as handle:
                response = client.audio.transcriptions.create(file=handle, **options)
            break
        except Exception as error:  # noqa: BLE001 - classified below
            # Out of credit, bad key, no model access: retrying only wastes time.
            reason = fatal_reason(error, model)
            if reason:
                _ABORT.set()
                raise FatalAPIError(reason) from error

            # A 400 usually means this model rejects one of the optional
            # parameters. Drop it and retry instead of failing every chunk -
            # model capabilities change faster than this script does.
            if getattr(error, "status_code", None) == 400:
                if adaptations < MAX_ADAPTATIONS:
                    # Plain json has no segments, so it would silently lose the speakers.
                    keep = {"response_format"} if "extra_body" in options else set()
                    dropped = drop_unsupported_option(options, error, keep=keep)
                    if dropped:
                        adaptations += 1
                        log(f"  {chunk.label} {model} rejected '{dropped}', retrying without it")
                        continue
                # Nothing left to drop: a 400 is deterministic, so retrying the
                # identical request just wastes time. Report what the API said.
                _ABORT.set()
                limit = duration_limit_from_error(error)
                advice = (f"  This model accepts at most {limit:g}s per request - "
                          f"re-run with --chunk {int(limit / CHUNK_MAX_FACTOR)}"
                          if limit else
                          "  If it is about audio length or size, re-run with a smaller --chunk.")
                raise FatalAPIError(
                    f"{model} rejected the request for {chunk.label} and retrying will not help.\n"
                    f"  API said: {api_message(error)}\n{advice}"
                ) from error

            # Too many requests in flight for the balance: wait for them to settle.
            # Not a failure, so it does not use up the chunk's real attempts.
            if is_in_flight_wait(error):
                inflight_waits += 1
                if inflight_waits > MAX_INFLIGHT_WAITS:
                    advice = ("Re-run with --workers 1, or add credits" if concurrent else
                              "Add credits or raise this key's limit")
                    raise RuntimeError(
                        f"{chunk.label}: OpenRouter kept refusing the request for this balance "
                        f"({api_message(error)}). {advice} at "
                        f"{PROVIDERS['openrouter']['credits_url']}") from error
                delay = retry_after_seconds(error) or 30.0
                log(f"  {chunk.label} waiting {delay:g}s: too many requests in flight "
                    "for the OpenRouter balance")
                _pause(delay)
                continue

            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{chunk.label} failed after {attempt} attempts: {api_message(error)}") from error
            delay = retry_after_seconds(error) or min(60, 4 * 2 ** (attempt - 1))
            log(f"  {chunk.label} attempt {attempt} failed "
                f"({type(error).__name__}: {api_message(error)[:120]}), retrying in {delay:g}s")
            _pause(delay)

    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    segments = payload_segments(payload, chunk.start, chunk.end)

    result = {
        "index": chunk.index,
        "start": chunk.start,
        "end": chunk.end,
        "text": (payload.get("text") or "").strip(),
        "segments": segments,
    }
    chunk.cache_path.parent.mkdir(parents=True, exist_ok=True)
    chunk.cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Output rendering
# --------------------------------------------------------------------------- #

def build_paragraphs(segments: list[dict], pause_threshold: float = 1.2,
                     max_chars: int = 900) -> list[str]:
    """Group timestamped segments into readable paragraphs on long pauses."""
    paragraphs: list[str] = []
    current: list[str] = []
    previous_end: float | None = None

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        gap = segment["start"] - previous_end if previous_end is not None else 0.0
        joined = " ".join(current)
        if current and (gap >= pause_threshold or len(joined) >= max_chars):
            paragraphs.append(joined)
            current = []
        current.append(text)
        previous_end = segment["end"]

    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def build_dialogue(segments: list[dict]) -> list[str]:
    """Group diarized segments into 'Speaker: line' blocks, merging consecutive turns.

    Turns are never merged across a chunk boundary: the model labels speakers
    independently per request, so 'A' in one chunk is not provably 'A' in the next.
    """
    lines: list[str] = []
    key: tuple | None = None
    speaker = "?"
    current: list[str] = []

    def flush() -> None:
        if current:
            lines.append(f"{speaker}: {' '.join(current)}")

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        who = segment.get("speaker") or "?"
        if (segment.get("chunk"), who) != key:
            flush()
            key, speaker, current = (segment.get("chunk"), who), who, []
        current.append(text)
    flush()
    return lines


def render_timestamped(segments: list[dict], block_seconds: float = 120.0) -> str:
    """Text with a [H:MM:SS] marker every `block_seconds`, for skimming long talks."""
    lines: list[str] = []
    block_start: float | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer and block_start is not None:
            lines.append(f"[{fmt_duration(block_start)}] " + " ".join(buffer))
            lines.append("")

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        if block_start is None:
            block_start = segment["start"]
        if segment["start"] - block_start >= block_seconds:
            flush()
            buffer, block_start = [], segment["start"]
        buffer.append(text)
    flush()
    return "\n".join(lines).strip() + "\n"


def render_dialogue_timestamped(segments: list[dict]) -> str:
    """One line per speaker turn, prefixed with its start time."""
    lines: list[str] = []
    key: tuple | None = None
    speaker = "?"
    start = 0.0
    current: list[str] = []

    def flush() -> None:
        if current:
            lines.append(f"[{fmt_duration(start)}] {speaker}: {' '.join(current)}")

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        who = segment.get("speaker") or "?"
        if (segment.get("chunk"), who) != key:
            flush()
            key, speaker, start, current = (segment.get("chunk"), who), who, segment["start"], []
        current.append(text)
    flush()
    return "\n".join(lines).strip() + "\n"


def render_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    counter = 0
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        if segment.get("speaker") is not None:
            text = f"{segment['speaker']}: {text}"
        counter += 1
        end = max(segment["end"], segment["start"] + 0.2)
        lines += [str(counter), f"{fmt_srt_time(segment['start'])} --> {fmt_srt_time(end)}", text, ""]
    return "\n".join(lines)


def rename_speakers(segments: list[dict], mapping: dict[str, str]) -> None:
    """Apply --speakers A=Denis,B=Client to diarized segments, in place."""
    for segment in segments:
        who = segment.get("speaker")
        if who and who in mapping:
            segment["speaker"] = mapping[who]


def write_outputs(output_dir: Path, stem: str, results: list[dict],
                  has_timestamps: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Tag every segment with its chunk so speaker turns are not merged across
    # a boundary, where the model's speaker labels restart from scratch.
    segments = [
        {**segment, "chunk": result.get("index")}
        for result in results for segment in result["segments"]
    ]
    written: list[Path] = []

    diarized = any(s.get("speaker") for s in segments)

    if diarized:
        body = "\n\n".join(build_dialogue(segments))
    elif has_timestamps and segments:
        body = "\n\n".join(build_paragraphs(segments))
    else:
        body = "\n\n".join(result["text"] for result in results if result["text"])

    plain = output_dir / f"{stem}.txt"
    plain.write_text(body.strip() + "\n", encoding="utf-8-sig")
    written.append(plain)

    if has_timestamps and segments:
        marked = output_dir / f"{stem}.timestamped.txt"
        marked.write_text(
            render_dialogue_timestamped(segments) if diarized else render_timestamped(segments),
            encoding="utf-8-sig",
        )
        written.append(marked)

        subtitles = output_dir / f"{stem}.srt"
        subtitles.write_text(render_srt(segments), encoding="utf-8")
        written.append(subtitles)

    return written


# --------------------------------------------------------------------------- #
# Per-file pipeline
# --------------------------------------------------------------------------- #

def process_file(path: Path, client: OpenAI, args: argparse.Namespace,
                 position: int = 1, total: int = 1) -> bool:
    stem = slugify(path.stem)
    final_txt = Path(args.output_dir) / f"{stem}.txt"
    if final_txt.exists() and not args.force:
        log(f"= {path.name}: already done ({final_txt.name}), skipping. Use --force to redo.")
        return True

    work_dir = Path(args.work_dir) / stem
    chunk_dir = work_dir / "chunks"

    spec = args.spec
    # Only a model with provider diarize options changes its request for --diarize
    # (gpt-4o-transcribe-diarize labels speakers anyway), so only then does the cache.
    diarize = bool(args.diarize and spec.diarize_options)
    duration = probe_duration(path)
    size_mb = path.stat().st_size / 1e6
    price = spec.price_per_minute
    cost = f", ~${duration / 60 * price:.2f}" if price else ""
    counter = f"[{position}/{total}] " if total > 1 else ""
    log(f"\n>> {counter}{path.name}  ({fmt_duration(duration)}, {size_mb:.1f} MB{cost})")

    show_progress = not args.no_progress
    silences = detect_silences(path, duration, show_progress)
    hard_max = spec.max_seconds - MODEL_LIMIT_MARGIN if spec.max_seconds else None
    spans = plan_cut_points(duration, silences, float(args.chunk), hard_max)
    log(f"   {len(silences)} pauses found -> {len(spans)} chunks of "
        f"~{fmt_duration(duration / max(1, len(spans)))}")
    if spec.diarizes and len(spans) > 1:
        # Said before any money is spent: the model labels each request on its own.
        log(f"   note: {len(spans)} requests needed, and speaker labels restart in each, "
            "so the same label in different parts may be a different person")

    if args.dry_run:
        for index, (start, end) in enumerate(spans):
            log(f"     [{index + 1:03d}] {fmt_duration(start)} - {fmt_duration(end)}")
        return True

    chunk_dir.mkdir(parents=True, exist_ok=True)
    bitrate = bitrate_for(args.chunk, spec.diarizes)
    signature = request_signature(args.model, args.language,
                                  args.prompt if spec.supports_prompt else None,
                                  diarize=diarize)
    chunks = []
    for index, (start, end) in enumerate(spans):
        audio_path, cache_path = chunk_paths(chunk_dir, index, start, end, signature, bitrate)
        chunks.append(Chunk(index, start, end, audio_path, cache_path))

    stale = prune_stale_audio(chunk_dir, {chunk.audio_path.name for chunk in chunks})
    if stale:
        log(f"   discarded {stale} chunk file(s) from an earlier cut plan")

    # Resume: anything already transcribed is loaded straight from cache, so it
    # costs nothing and is excluded from the progress totals below.
    results: dict[int, dict] = {}
    pending: list[Chunk] = []
    for chunk in chunks:
        cached = read_cache(chunk)
        if cached is None:
            pending.append(chunk)
        else:
            results[chunk.index] = cached
    if results:
        log(f"   {len(results)}/{len(chunks)} chunks restored from cache ({chunk_dir})")
    else:
        foreign = foreign_cache_count(chunk_dir, signature)
        if foreign:
            log(f"   note: {foreign} chunk(s) in {chunk_dir} were transcribed with other settings "
                "(model, language, prompt or --diarize); re-run with those to reuse them free")

    if pending:
        with Progress(len(pending), "cut chunks  ", enabled=show_progress) as bar:
            for step, chunk in enumerate(pending, start=1):
                bar.set_note(f"{step}/{len(pending)}")
                if not (chunk.audio_path.is_file() and chunk.audio_path.stat().st_size > 2048):
                    extract_chunk(path, chunk.audio_path, chunk.start, chunk.end, bitrate)
                bar.advance(1, note=f"{step}/{len(pending)}")

    started = time.monotonic()
    pending_seconds = sum(chunk.end - chunk.start for chunk in pending) or 1.0
    done = 0

    if pending:
        parallel = min(args.workers, len(pending))
        log(f"   sending {len(pending)} chunk(s) to {args.model}, {parallel} at a time - "
            "a long chunk can take a few minutes, the bar keeps counting")

    # ETA is weighted by audio seconds, so uneven chunk lengths do not skew it.
    inflight = 0
    counter_lock = threading.Lock()

    with Progress(pending_seconds, "transcribe  ", enabled=show_progress and bool(pending)) as bar:

        def status() -> str:
            waiting = f", {inflight} in flight" if inflight else ""
            return f"{done}/{len(pending)} chunks{waiting}"

        bar.set_value(0, note=status())

        def finish(chunk: Chunk, result: dict) -> None:
            nonlocal done
            done += 1
            results[chunk.index] = result
            bar.advance(chunk.end - chunk.start, note=status())

        def run(chunk: Chunk, prompt: str | None) -> dict:
            nonlocal inflight
            with counter_lock:
                inflight += 1
            bar.set_note(status())
            try:
                return transcribe_chunk(client, chunk, args.model, args.language, prompt,
                                        diarize=diarize,
                                        concurrent=min(args.workers, len(pending)) > 1)
            finally:
                with counter_lock:
                    inflight -= 1

        if args.workers <= 1:
            # Sequential mode: feed the tail of the previous chunk forward as context.
            context = args.prompt or ""
            for chunk in pending:
                result = run(chunk, context or None)
                context = (result["text"] or context)[-PROMPT_CONTEXT_CHARS:]
                finish(chunk, result)
        elif pending:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run, chunk, args.prompt): chunk for chunk in pending}
                for future in concurrent.futures.as_completed(futures):
                    finish(futures[future], future.result())

    ordered = [results[index] for index in sorted(results)]
    if args.speakers:
        for result in ordered:
            rename_speakers(result["segments"], args.speakers)

    written = write_outputs(Path(args.output_dir), stem, ordered, spec.timestamps)

    words = sum(len(result["text"].split()) for result in ordered)
    log(f"   transcribed in {fmt_short(time.monotonic() - started)}, ~{words} words")
    if spec.diarizes:
        voices = sorted({s["speaker"] for r in ordered for s in r["segments"] if s.get("speaker")})
        log(f"   speakers: {', '.join(voices) if voices else 'none detected'}")
        if voices and not args.speakers:
            example = ",".join(f"{voice}=Name" for voice in voices[:2])
            log(f"   tip: name them with --speakers {example}")
    for out in written:
        log(f"   -> {out}")
    if not spec.timestamps:
        log(f"   note: {args.model} returns no timestamps, so only .txt was produced "
            "(use --model mai or whisper-1 for .srt subtitles)")

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def collect_inputs(paths: list[str], input_dir: Path) -> list[Path]:
    if paths:
        files: list[Path] = []
        for raw in paths:
            candidate = Path(raw)
            if not candidate.is_absolute() and not candidate.exists():
                candidate = input_dir / raw
            if candidate.is_dir():
                files += sorted(p for p in candidate.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
            elif candidate.is_file():
                files.append(candidate)
            else:
                log(f"! not found: {raw}")
        return files

    if not input_dir.is_dir():
        input_dir.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe long audio recordings with OpenRouter or OpenAI speech-to-text models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="specific files or folders (default: everything in ./input)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="transcription model: an alias (mai, voxtral, scribe), a built-in "
                             "id, or any OpenRouter speech-to-text id; run --list-models")
    parser.add_argument("--list-models", action="store_true",
                        help="print what each model supports and exit")
    parser.add_argument("--language", default="ru",
                        help="ISO-639-1 code of the spoken language, or 'auto' to let the model decide")
    parser.add_argument("--prompt", default=None,
                        help="hint with names/terms/spelling to improve accuracy")
    parser.add_argument("--diarize", action="store_true",
                        help="label who is speaking (with --model mai; the whole recording goes "
                             "in one request when it fits, so labels stay consistent)")
    parser.add_argument("--speakers", default=None, metavar="0=Name,1=Name",
                        help="rename speaker labels in the output (mai: 0,1,...; "
                             "gpt-4o-transcribe-diarize: A,B,...)")
    parser.add_argument("--chunk", type=int, metavar="SECONDS", default=None,
                        help="override the target chunk length; without it each model's own "
                             "default is used (900 s; 4792 s with --diarize), lowered further "
                             "if the model caps a single request")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel API requests; 1 = sequential (OpenAI models then also carry "
                             "the previous chunk forward as context)")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR),
                        help="where to look for recordings")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="where to write the transcripts")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help="scratch space for chunk audio and the resume cache")
    parser.add_argument("--force", action="store_true", help="re-transcribe even if output already exists")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep chunk audio and per-chunk JSON cache after success")
    parser.add_argument("--dry-run", action="store_true",
                        help="only show the chunk plan, make no API calls")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable the progress bars (plain log lines only)")

    args = parser.parse_args(argv)
    if not args.list_models:
        try:
            args.model = resolve_model(args.model)
        except ValueError as error:
            parser.error(str(error))
    elif args.model not in MODEL_SPECS:
        args.model = DEFAULT_MODEL      # --list-models never needs the chosen model
    if args.language and args.language.lower() in {"auto", "", "none"}:
        args.language = None
    args.workers = max(1, args.workers)

    spec = MODEL_SPECS[args.model]
    if args.diarize and not spec.diarizes:
        if spec.diarize_options is None:
            capable = [alias for alias, model in MODEL_ALIASES.items()
                       if MODEL_SPECS[model].diarize_options]
            parser.error(f"--diarize is not available for {args.model}; use --model "
                         f"{' / '.join(capable)} (or --model gpt-4o-transcribe-diarize)")
        # One request per recording while it fits: speaker labels then stay
        # consistent across the whole file instead of restarting every chunk.
        spec = dataclasses.replace(spec, diarizes=True,
                                   default_chunk_seconds=DIARIZE_CHUNK_SECONDS,
                                   max_seconds=DIARIZE_MAX_SECONDS)
    args.spec = spec          # the effective spec for this run; MODEL_SPECS stays as built

    explicit_chunk = args.chunk is not None
    args.chunk = max(60, args.chunk) if args.chunk else spec.default_chunk_seconds
    if spec.max_seconds:
        # Keep target x CHUNK_MAX_FACTOR inside the model's per-request limit.
        allowed = int((spec.max_seconds - MODEL_LIMIT_MARGIN) / CHUNK_MAX_FACTOR)
        if args.chunk > allowed:
            if explicit_chunk:
                log(f"note: {args.model} accepts at most {spec.max_seconds:g}s of audio per "
                    f"request, lowering --chunk from {args.chunk} to {allowed}")
            args.chunk = allowed

    mapping: dict[str, str] = {}
    for pair in (args.speakers or "").split(","):
        label, _, name = pair.partition("=")
        if label.strip() and name.strip():
            mapping[label.strip()] = name.strip()
    if args.speakers and not mapping:
        parser.error("--speakers expects pairs like  0=Denis,1=Client  (mai --diarize) "
                     "or  A=Denis,B=Client  (gpt-4o-transcribe-diarize)")
    if mapping and not spec.diarizes:
        if spec.diarize_options is not None:
            parser.error(f"--speakers needs speaker labels: add --diarize "
                         f"(e.g. --diarize --speakers 0=Name,1=Name)")
        parser.error(f"--speakers only works with a diarizing model, and {args.model} is not one "
                     "(use --model mai --diarize, or --model gpt-4o-transcribe-diarize)")
    args.speakers = mapping
    return args


def speakers_cell(spec: ModelSpec) -> str:
    """'yes' if the model always labels speakers, '--diarize' if it can on request."""
    if spec.diarizes:
        return "yes"
    return "--diarize" if spec.diarize_options else "no"


def print_model_table() -> None:
    """Compare the built-in transcription models."""
    log("Built-in speech-to-text models:\n")
    aliases = {model: alias for alias, model in MODEL_ALIASES.items()}
    header = (f"  {'model':39} {'alias':8} {'via':11} {'timestamps':11} {'speakers':9} "
              f"{'prompt':7} {'$/min':8} note")
    log(header)
    log("  " + "-" * (len(header) - 2))
    for name, spec in MODEL_SPECS.items():
        price = f"${spec.price_per_minute:.4g}" if spec.price_per_minute else "?"
        log(f"  {name:39} {aliases.get(name, ''):8} {spec.provider:11} "
            f"{'yes' if spec.timestamps else 'no':11} "
            f"{speakers_cell(spec):9} {'yes' if spec.supports_prompt else 'no':7} "
            f"{price:8} {spec.note}")
    log(f"\n  default    -> {DEFAULT_MODEL}")
    log("  via        -> openai needs OPENAI_API_KEY, openrouter needs OPENROUTER_API_KEY")
    log("  timestamps -> needed for .srt and .timestamped.txt output")
    log("  speakers   -> diarization: labels each turn with who said it "
        "(A, B, ... or 0, 1, ... with --diarize)")
    log("  prices: OpenAI per https://developers.openai.com/api/docs/pricing (2026-07-30),")
    log("          OpenRouter per its models API (2026-11-17)")
    log("  any other OpenRouter model works by its full id, e.g. --model x-ai/grok-stt-2.0;")
    log("  browse them at https://openrouter.ai/collections/speech-to-text-models")


def other_provider_hint(provider: str, diarize: bool = False) -> str:
    """Point at a model the user can already run, when only the other key is set."""
    if provider == "openrouter" and os.environ.get(PROVIDERS["openai"]["key_env"], "").strip():
        if diarize:
            return ("\n  OPENAI_API_KEY is set, so you can label speakers with OpenAI instead: "
                    "--model gpt-4o-transcribe-diarize")
        return ("\n  OPENAI_API_KEY is set, so you can keep using OpenAI instead: "
                "--model whisper-1 (the previous default)")
    return ""


class MissingKeyError(RuntimeError):
    """The API key for the chosen model's provider is not configured."""


def make_client(provider: str, diarize: bool = False) -> OpenAI:
    """An SDK client for `provider`, keyed from the environment or .env."""
    config = PROVIDERS[provider]
    load_dotenv(ROOT / ".env")
    key_env = config["key_env"]
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise MissingKeyError(
            f"{key_env} is not set (needed for {provider} models). Get a key at {config['key_url']}\n"
            f"  PowerShell (this session):  $env:{key_env} = '...'\n"
            f"  or put {key_env}=... into .env (see .env.example)."
            + other_provider_hint(provider, diarize))
    # max_retries=1: this script does its own classified retries, and the
    # SDK's blind ones only add delay before a fatal error surfaces.
    # The generous timeout is for diarization, which is slow to answer.
    return OpenAI(api_key=api_key, base_url=config["base_url"],
                  max_retries=1, timeout=REQUEST_TIMEOUT)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    if args.list_models:
        print_model_table()
        return 0

    require_tool("ffmpeg")
    require_tool("ffprobe")

    files = collect_inputs(args.files, Path(args.input_dir))
    if not files:
        log(f"No audio files found. Put recordings into {args.input_dir} and run again.\n"
            f"Recognised extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return 0

    spec = args.spec
    client = None
    if not args.dry_run:
        try:
            client = make_client(spec.provider, diarize=args.diarize)
        except MissingKeyError as error:
            log(str(error))
            return 2

    log(f"Model: {args.model} via {spec.provider} | language: {args.language or 'auto'} | "
        f"chunk: {args.chunk}s | workers: {args.workers} | files: {len(files)}")
    if args.prompt and not spec.supports_prompt:
        log(f"   warning: {args.model} does not accept --prompt, it will be ignored")

    failures = 0
    for position, path in enumerate(files, start=1):
        try:
            process_file(path, client, args, position, len(files))
        except FatalAPIError as error:
            if _ACTIVE_BAR is not None:
                _ACTIVE_BAR.close(completed=False)
            log(f"\n! Stopped: {error}")
            return 3
        except KeyboardInterrupt:
            if _ACTIVE_BAR is not None:
                _ACTIVE_BAR.close(completed=False)
            log("\nInterrupted. Finished chunks are cached in work/, re-run to continue.")
            return 130
        except Exception as error:  # noqa: BLE001 - keep going with the remaining files
            failures += 1
            log(f"! {path.name} failed: {error}")

    if failures:
        log(f"\nCompleted with {failures} failed file(s). Cached chunks are kept in {args.work_dir} - "
            "re-running resumes where it stopped.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
