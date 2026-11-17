# Diarize via MAI + Review Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--diarize` labels speakers with MAI-Transcribe 2 in one request per recording, plus the 8 deferred review fixes.

**Architecture:** `ModelSpec.diarize_options` + `--diarize` swaps in a diarizing spec (big chunk, 32k); provider options travel as a JSON string in `extra_body`; speaker `0` handled; cache signature includes the mode.

**Tech Stack:** Python 3.10, openai 2.x SDK, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-diarize-and-polish-design.md`

## Global Constraints

- MAI diarize options: `{"azure": {"diarization": {"enabled": True}}}`, sent as `extra_body={"provider": json.dumps({"options": <opts>})}`.
- Diarizing spec: `diarizes=True, default_chunk_seconds=4800, max_seconds=6000`.
- In-flight 402 wait limit: `MAX_INFLIGHT_WAITS = 10`.
- No new dependencies.

## Review Focus

- `--diarize` with a model that can't (`--model voxtral --diarize`) → parser error naming `mai`. Test in Task 1.
- Short recording run once without and once with `--diarize` → second run must not reuse the first cache. Test in Task 1.
- Speaker `0` (int) must survive into `.txt`/`.srt` and `--speakers 0=...` must rename it. Test in Task 1.
- `--speakers` without `--diarize` on mai → error that suggests `--diarize`. Test in Task 1.
- Tests leave `MODEL_SPECS` as they found it. Fixture in Task 4.

---

### Task 1: `--diarize` for MAI

**Files:** Modify `transcribe.py` (ModelSpec, MAI entry, request_signature, payload_segments, render_srt, transcribe_chunk, process_file, parse_args, `--speakers` help). Test: `tests/test_diarize.py`.

**Interfaces:** Produces `ModelSpec.diarize_options: dict | None = None`; `diarize_extra_body(options: dict) -> dict`; `request_signature(model, language, prompt, diarize: bool = False) -> str`; `transcribe_chunk(..., diarize: bool = False)`.

- [ ] **Step 1: Failing tests**

```python
def test_mai_has_diarize_options():
    assert t.MODEL_SPECS["microsoft/mai-transcribe-2"].diarize_options == {"azure": {"diarization": {"enabled": True}}}

def test_extra_body_is_json_string():
    body = t.diarize_extra_body({"azure": {"diarization": {"enabled": True}}})
    assert json.loads(body["provider"]) == {"options": {"azure": {"diarization": {"enabled": True}}}}

def test_diarize_flag_switches_spec():
    args = t.parse_args(["--diarize"])
    spec = t.MODEL_SPECS[args.model]
    assert args.diarize and spec.diarizes and spec.default_chunk_seconds == 4800 and spec.max_seconds == 6000
    assert args.chunk <= int((6000 - t.MODEL_LIMIT_MARGIN) / t.CHUNK_MAX_FACTOR)

def test_diarize_rejected_for_unsupported_model():
    with pytest.raises(SystemExit): t.parse_args(["--model", "voxtral", "--diarize"])

def test_speakers_without_diarize_suggests_flag(capsys):
    with pytest.raises(SystemExit): t.parse_args(["--speakers", "0=A"])
    assert "--diarize" in capsys.readouterr().err

def test_speakers_with_numeric_labels():
    assert t.parse_args(["--diarize", "--speakers", "0=Мама,1=Консультант"]).speakers == {"0": "Мама", "1": "Консультант"}

def test_signature_depends_on_diarize():
    assert t.request_signature("m", "ru", None) != t.request_signature("m", "ru", None, diarize=True)

def test_speaker_zero_kept():
    segs = t.payload_segments({"segments": [{"start": 0, "end": 1, "text": "a", "speaker": 0},
                                            {"start": 1, "end": 2, "text": "b", "speaker": 1}]}, 0)
    assert [s["speaker"] for s in segs] == ["0", "1"]

def test_srt_shows_speaker_zero():
    assert "Мама: a" in t.render_srt([{"start": 0, "end": 1, "text": "a", "speaker": "Мама"}])
    assert "0: a" in t.render_srt([{"start": 0, "end": 1, "text": "a", "speaker": "0"}])
```

- [ ] **Step 2:** `python -m pytest tests/test_diarize.py -q` → FAIL.
- [ ] **Step 3: Implement.**
  - `parse_args`: `--diarize` flag (store_true).
  - Speaker-name check: error "add --diarize" when the model has `diarize_options`; the existing gpt-4o-transcribe-diarize hint otherwise.
  - `transcribe_chunk` adds `extra_body` when `diarize and spec.diarize_options`.
  - `process_file` passes `args.diarize` to the signature and to `transcribe_chunk`.
- [ ] **Step 4:** full suite green. **Step 5:** commit `feat: --diarize labels speakers with MAI-Transcribe 2`.

### Task 2: Error polish (review items 3, 4)

**Files:** `transcribe.py` (`fatal_reason`, `transcribe_chunk` retry loop). Test: `tests/test_errors.py` (append).

- [ ] **Step 1: Failing tests**

```python
def test_402_text_names_openrouter_not_model():
    msg = t.fatal_reason(FakeError(402, 402, {"message": "Insufficient credits", "code": 402}), MAI)
    assert "OpenRouter" in msg and "microsoft" not in msg

def test_403_includes_api_message():
    assert "flagged by moderation" in t.fatal_reason(FakeError(403, 403, {"message": "flagged by moderation"}), MAI)

def test_in_flight_402_detector():
    body = {"message": "x", "code": 402, "metadata": {"reason": "in_flight_budget_exhausted"}}
    assert t.is_in_flight_wait(FakeError(402, 402, body))
    assert not t.is_in_flight_wait(FakeError(402, 402, {"message": "x"}))
```

- [ ] **Step 2:** FAIL. **Step 3: Implement.**
  - `is_in_flight_wait(error) -> bool`.
  - In the retry loop an in-flight 402 increments `inflight_waits`, not `attempt`.
  - Past `MAX_INFLIGHT_WAITS`, raise `RuntimeError` mentioning `--workers 1`.
- [ ] **Step 4:** suite green. **Step 5:** commit `fix: clearer OpenRouter 402/403, in-flight waits do not burn attempts`.

### Task 3: CLI polish (review items 1, 2, 5, 6)

**Files:** `transcribe.py` (`make_client`, `process_file`, `resolve_model`). Test: `tests/test_cli.py`, `tests/test_models.py` (append).

- [ ] **Step 1: Failing tests**

```python
def test_missing_openrouter_key_suggests_whisper_when_openai_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test"); monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(t, "load_dotenv", lambda p: None)
    with pytest.raises(t.MissingKeyError, match="--model whisper-1"): t.make_client("openrouter")

def test_plain_name_typo_suggests(): 
    with pytest.raises(ValueError, match="scribe"): t.resolve_model("scirbe")

def test_catalog_any_failure_is_a_warning():
    def broken(): raise KeyError("id")
    assert t.resolve_model("acme/other-stt", catalog_loader=broken) == "acme/other-stt"

def test_foreign_cache_count(tmp_path):
    (tmp_path / "000_000000000-000060000_aaaaaaaa.json").write_text("{}")
    (tmp_path / "000_000000000-000060000_bbbbbbbb.json").write_text("{}")
    assert t.foreign_cache_count(tmp_path, "aaaaaaaa") == 1
```

- [ ] **Step 2:** FAIL. **Step 3: Implement.**
  - `foreign_cache_count(chunk_dir: Path, signature: str) -> int`. `process_file` logs a hint when it is > 0 and no chunk was restored.
  - `resolve_model` suggestions via `difflib.get_close_matches(name, [*MODEL_ALIASES, *MODEL_SPECS])`.
  - Catalog: `except Exception`.
- [ ] **Step 4:** suite green. **Step 5:** commit `fix: friendlier hints for keys, typos, old cache, catalog errors`.

### Task 4: Test isolation, docs, end-to-end

**Files:** `tests/conftest.py`, `README.md`, `README.ru.md`, `transcribe.py` (`--workers` help).

- [ ] **Step 1:** autouse fixture in `conftest.py`: copy `t.MODEL_SPECS` before each test, restore after. Suite still green.
- [ ] **Step 2: Docs.**
  - `--workers` help: "1 = sequential (OpenAI models also carry context forward)".
  - README sample cost → MAI price (2:47 ≈ $0.28).
  - Options table: `--diarize`; `--workers` wording.
  - Diarization section: `--diarize` first (MAI, whole recording in one request, labels 0/1, `--speakers 0=…`); gpt-4o-transcribe-diarize as the OpenAI alternative.
- [ ] **Step 3: Real runs.**
  - `python transcribe.py <3-min dialog mp3> --diarize --speakers 0=A,1=B --output-dir <tmp>` → `.txt` is a dialogue with names, `.srt` has names.
  - `python transcribe.py <40-min combined ogg> --diarize --output-dir <tmp>` → 1 chunk, two speakers.
- [ ] **Step 4:** commit `docs: --diarize`; push; PR; merge.
