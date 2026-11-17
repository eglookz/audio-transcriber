# OpenRouter Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `transcribe.py` use OpenRouter STT models (MAI-Transcribe 2, Voxtral Small 3, Scribe v2, or any OpenRouter id) next to the existing OpenAI ones, picked with `--model`.

**Architecture:** `ModelSpec` gains a `provider`; the OpenAI SDK client is built per provider (`base_url` + key). Aliases and unknown `vendor/model` ids are resolved in `parse_args` and registered into `MODEL_SPECS`, so the rest of the pipeline is unchanged. Responses that carry only word timestamps are turned into segments.

**Tech Stack:** Python 3.10, `openai` 2.x SDK (multipart to `https://openrouter.ai/api/v1`), stdlib `urllib` for the model catalog, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-openrouter-models-design.md`

## Global Constraints

- No new dependencies in `requirements.txt` (catalog fetch uses `urllib.request`).
- `DEFAULT_MODEL = "microsoft/mai-transcribe-2"`.
- Aliases exactly: `mai` → `microsoft/mai-transcribe-2`, `voxtral` → `mistralai/voxtral-small-transcribe-3-0`, `scribe` → `elevenlabs/scribe-v2`.
- Prices ($/min): MAI 0.00167, Voxtral 0.004, Scribe 0.00367.
- Keys: `OPENAI_API_KEY` for provider `openai`, `OPENROUTER_API_KEY` for `openrouter`; env var or `.env`.
- Existing OpenAI models, `--prompt`, `--speakers`, cache file names: behaviour unchanged.

## Review Focus

- Typo in a full id (`microsoft/mai-transcribe-3`) → error before any API call, with close matches. Test in Task 1.
- Catalog unreachable (offline) for an unknown id → warning, run continues with generic spec. Test in Task 1.
- OpenRouter error with integer `code` (402/400) must not crash `fatal_reason`. Test in Task 3.
- Only `OPENAI_API_KEY` set but default MAI model → clear message naming `OPENROUTER_API_KEY`. Test in Task 2.
- Scribe chunk with an empty `words` list / single segment → no crash, falls back to text. Test in Task 4.

---

### Task 1: Model registry — provider, winners, aliases, any OpenRouter id

**Files:**
- Modify: `transcribe.py` (`ModelSpec`, `MODEL_SPECS`, `DEFAULT_MODEL`, new helpers after `MODEL_SPECS`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `ModelSpec.provider: str = "openai"`, `ModelSpec.word_timestamps: bool = False`;
  `MODEL_ALIASES: dict[str, str]`;
  `fetch_openrouter_catalog(timeout: float = 20.0) -> dict[str, float | None]` (id → $/min, raises `OSError` on network failure);
  `resolve_model(name: str, catalog_loader=fetch_openrouter_catalog) -> str` — returns the full id, registering a generic OpenRouter spec in `MODEL_SPECS` for unknown `x/y` ids; raises `ValueError` with a helpful message for unknown names.

- [ ] **Step 1: Write failing tests**

```python
import transcribe as t

def test_aliases_resolve_to_full_ids():
    assert t.resolve_model("mai") == "microsoft/mai-transcribe-2"
    assert t.resolve_model("voxtral") == "mistralai/voxtral-small-transcribe-3-0"
    assert t.resolve_model("scribe") == "elevenlabs/scribe-v2"
    assert t.resolve_model("whisper-1") == "whisper-1"

def test_winners_are_openrouter_with_timestamps():
    for alias in ("mai", "voxtral", "scribe"):
        spec = t.MODEL_SPECS[t.resolve_model(alias)]
        assert spec.provider == "openrouter" and spec.timestamps and not spec.supports_prompt
    assert t.MODEL_SPECS["elevenlabs/scribe-v2"].word_timestamps
    assert t.MODEL_SPECS["whisper-1"].provider == "openai"
    assert t.DEFAULT_MODEL == "microsoft/mai-transcribe-2"

def test_unknown_openrouter_id_is_registered_with_catalog_price():
    model = t.resolve_model("x-ai/grok-stt-2.0", catalog_loader=lambda: {"x-ai/grok-stt-2.0": 0.00167})
    spec = t.MODEL_SPECS[model]
    assert spec.provider == "openrouter" and spec.price_per_minute == 0.00167
    assert spec.response_format == "verbose_json"

def test_typo_in_openrouter_id_fails_with_suggestion():
    with pytest.raises(ValueError, match="microsoft/mai-transcribe-2"):
        t.resolve_model("microsoft/mai-transcribe-3",
                        catalog_loader=lambda: {"microsoft/mai-transcribe-2": 0.00167})

def test_catalog_offline_still_accepts_id():
    def offline(): raise OSError("no network")
    model = t.resolve_model("acme/new-stt", catalog_loader=offline)
    assert t.MODEL_SPECS[model].price_per_minute is None

def test_unknown_plain_name_fails():
    with pytest.raises(ValueError, match="--list-models"):
        t.resolve_model("whisper-9")
```

- [ ] **Step 2: Run** `python -m pytest tests/test_models.py -q` — expect FAIL (`resolve_model` missing).
- [ ] **Step 3: Implement.** Add fields to `ModelSpec`; add the three winners to `MODEL_SPECS` (all: `response_format="verbose_json"`, `timestamps=True`, `supports_prompt=False`, `supports_temperature=True`, `provider="openrouter"`, prices above; Scribe `word_timestamps=True`); `MODEL_ALIASES`; `fetch_openrouter_catalog` reads `GET https://openrouter.ai/api/v1/models?output_modalities=transcription`, price per minute = `float(pricing.prompt) * 60` only when `pricing.completion` is absent/zero (token-priced models → `None`); `resolve_model` uses `difflib.get_close_matches` for suggestions. Catalog lookup is skipped for ids already in `MODEL_SPECS`.
- [ ] **Step 4: Run** tests — expect PASS.
- [ ] **Step 5: Commit** `feat: register OpenRouter STT models and aliases`.

### Task 2: Per-provider client, keys and CLI

**Files:**
- Modify: `transcribe.py` (`parse_args`, `main`, `print_model_table`, module docstring)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `resolve_model`, `ModelSpec.provider`.
- Produces: `PROVIDERS: dict[str, dict]` with `base_url`, `key_env`, `key_url` per provider; `make_client(provider: str) -> OpenAI` (raises `MissingKeyError(RuntimeError)` whose message names the env var and `key_url`).

- [ ] **Step 1: Write failing tests**

```python
def test_parse_args_resolves_alias():
    assert t.parse_args(["--model", "voxtral"]).model == "mistralai/voxtral-small-transcribe-3-0"

def test_parse_args_default_is_mai():
    assert t.parse_args([]).model == "microsoft/mai-transcribe-2"

def test_parse_args_rejects_unknown_model(capsys):
    with pytest.raises(SystemExit):
        t.parse_args(["--model", "whisper-9"])

def test_missing_openrouter_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    with pytest.raises(t.MissingKeyError, match="OPENROUTER_API_KEY"):
        t.make_client("openrouter")

def test_openrouter_client_points_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert str(t.make_client("openrouter").base_url).startswith("https://openrouter.ai/api/v1")
```

- [ ] **Step 2: Run** `python -m pytest tests/test_cli.py -q` — expect FAIL.
- [ ] **Step 3: Implement.** Drop `choices=` from `--model`; in `parse_args` call `resolve_model` and turn `ValueError` into `parser.error`. `make_client` loads `.env`, reads the provider's key, builds `OpenAI(api_key=..., base_url=..., max_retries=1, timeout=REQUEST_TIMEOUT)`. `main` uses it and prints `MissingKeyError` text (exit 2). `print_model_table` gains a provider column, lists aliases, and ends with a line pointing to `https://openrouter.ai/collections/speech-to-text-models` for other ids. Description strings stop saying "OpenAI Audio API" only.
- [ ] **Step 4: Run** all tests — PASS.
- [ ] **Step 5: Commit** `feat: pick OpenAI or OpenRouter client from the chosen model`.

### Task 3: OpenRouter-aware error handling

**Files:**
- Modify: `transcribe.py` (`FATAL_HINTS`, `fatal_reason`, `retry_after_seconds`)
- Test: `tests/test_errors.py`

**Interfaces:**
- Consumes: `MODEL_SPECS[model].provider`, `PROVIDERS`.
- Produces: `fatal_reason(error, model) -> str | None` unchanged signature; tolerates non-string `code`.

- [ ] **Step 1: Write failing tests** using a tiny fake: `class FakeError(Exception)` with `status_code`, `code`, `body`, `response=None`.

```python
def err(status, code=None, body=None): ...

def test_in_flight_402_is_retryable():
    body = {"message": "would exceed", "code": 402, "metadata": {"reason": "in_flight_budget_exhausted",
            "headers": {"Retry-After": "120"}}}
    e = err(402, 402, body)
    assert t.fatal_reason(e, "microsoft/mai-transcribe-2") is None
    assert t.retry_after_seconds(e) == 120

def test_other_402_is_fatal_with_credits_link():
    e = err(402, 402, {"message": "Insufficient credits", "code": 402})
    assert "openrouter.ai/settings/credits" in t.fatal_reason(e, "microsoft/mai-transcribe-2")

def test_integer_code_does_not_crash():
    assert t.fatal_reason(err(500, 500, {"code": 500}), "microsoft/mai-transcribe-2") is None

def test_401_names_the_right_key():
    assert "OPENROUTER_API_KEY" in t.fatal_reason(err(401, 401), "microsoft/mai-transcribe-2")
    assert "OPENAI_API_KEY" in t.fatal_reason(err(401, "invalid_api_key"), "whisper-1")
```

- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement.** `code = str(getattr(error, "code", None) or "").strip()`; 402 → retryable when `body.metadata.reason == "in_flight_budget_exhausted"`, else fatal (OpenRouter credits hint). 401/404 hints use the provider's key name / model-list advice. `retry_after_seconds` falls back to `body["metadata"]["headers"]["Retry-After"]`.
- [ ] **Step 4: Run** all tests — PASS.
- [ ] **Step 5: Commit** `fix: classify OpenRouter errors, retry in-flight 402`.

### Task 4: Segments from word timestamps (Scribe)

**Files:**
- Modify: `transcribe.py` (`transcribe_chunk` request options + payload parsing; new `segments_from_words`)
- Test: `tests/test_words.py`

**Interfaces:**
- Produces: `segments_from_words(words: list[dict], offset: float) -> list[dict]` → `[{"start", "end", "text"}]`, absolute times (`+ offset`).
- Constants: `WORD_PAUSE_SPLIT = 0.8` s, `WORD_SEGMENT_SOFT_MAX = 15.0` s (split after `.?!…`), `WORD_SEGMENT_HARD_MAX = 30.0` s.

- [ ] **Step 1: Write failing tests**

```python
def w(word, start, end): return {"word": word, "start": start, "end": end}

def test_split_on_pause_and_offset():
    segs = t.segments_from_words([w("Привет,", 0, .5), w("мир.", .6, 1), w("Дальше", 2.5, 3)], offset=100)
    assert [s["text"] for s in segs] == ["Привет, мир.", "Дальше"]
    assert segs[0]["start"] == 100 and segs[1]["end"] == 103

def test_long_run_splits_after_sentence_end():
    words = [w(f"слово{i}.", i * 0.5, i * 0.5 + 0.4) for i in range(40)]  # 20 s, no pauses
    segs = t.segments_from_words(words, 0)
    assert len(segs) >= 2 and all(s["end"] - s["start"] <= 30 for s in segs)

def test_empty_words():
    assert t.segments_from_words([], 0) == []
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `segments_from_words`. In `transcribe_chunk`: when `spec.word_timestamps`, add `options["timestamp_granularities"] = ["word", "segment"]`; after the call, if `payload["words"]` is non-empty and the payload has ≤ 1 segment, use `segments_from_words(words, chunk.start)` instead of the segment loop.
- [ ] **Step 4: Run** all tests — PASS.
- [ ] **Step 5: Commit** `feat: build segments from word timestamps`.

### Task 5: Docs and end-to-end check

**Files:**
- Modify: `README.md`, `README.ru.md`, `.env.example`

- [ ] **Step 1:** `.env.example` gets `OPENROUTER_API_KEY=sk-or-...` with the keys link. READMEs: intro mentions OpenRouter; install shows both keys; usage shows `--model mai|voxtral|scribe`, full-id form, `--model whisper-1` for the old behaviour; a short comparison table (prices + one-line verdicts from the spec); note that `--prompt` is ignored by OpenRouter models.
- [ ] **Step 2: Real run.** Cut a 90 s clip (`ffmpeg -ss 300 -t 90 -i input/<ogg> work-e2e/clip.mp3`, outside the repo or under ignored `work/`) and run `python transcribe.py <clip> --output-dir <tmp> --force --model X` for `mai`, `voxtral`, `scribe`, and `x-ai/grok-stt-2.0`. Expected: each writes `.txt`, `.timestamped.txt`, `.srt`; Scribe's `.srt` has many cues; MAI cost line printed. `python transcribe.py --list-models` and `--dry-run` work without keys.
- [ ] **Step 3:** Full test suite green: `python -m pytest -q`.
- [ ] **Step 4: Commit** `docs: OpenRouter models in README` and push the branch.
