"""Deferred minors from the --diarize branch review."""
import time

import pytest

import transcribe as t
from test_errors import FakeError

MAI = "microsoft/mai-transcribe-2"


def run_process_file(tmp_path, monkeypatch, argv, duration=60.0, logs=None):
    """process_file with ffmpeg and the API stubbed out; returns the transcribe_chunk calls."""
    audio = tmp_path / "talk.mp3"
    audio.write_bytes(b"x" * 4096)
    calls = []
    monkeypatch.setattr(t, "probe_duration", lambda path: duration)
    monkeypatch.setattr(t, "detect_silences", lambda path, d, show: [])
    monkeypatch.setattr(t, "extract_chunk", lambda src, dst, s, e, bitrate="48k": dst.write_bytes(b"x" * 4096))
    monkeypatch.setattr(t, "log", (logs.append if logs is not None else (lambda m: None)))

    def fake_transcribe(client, chunk, model, language, prompt, diarize=False, concurrent=True):
        calls.append((chunk.cache_path.name, diarize))
        return {"index": chunk.index, "start": chunk.start, "end": chunk.end,
                "text": "a", "segments": [{"start": chunk.start, "end": chunk.end, "text": "a"}]}

    monkeypatch.setattr(t, "transcribe_chunk", fake_transcribe)
    args = t.parse_args([str(audio), "--output-dir", str(tmp_path / "out"),
                         "--work-dir", str(tmp_path / "w"), "--no-progress", "--force", *argv])
    t.process_file(audio, None, args)
    return calls


# --- parse_args no longer rewrites the global registry -----------------------

def test_diarize_does_not_mutate_registry():
    args = t.parse_args(["--diarize"])
    assert args.spec.diarizes
    assert not t.MODEL_SPECS[MAI].diarizes
    assert t.parse_args([]).chunk == 900


def test_list_models_points_mai_to_diarize(capsys):
    t.print_model_table()
    line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith(MAI))
    assert "--diarize" in line


# --- cache signature only changes when the request changes -------------------

def test_openai_diarize_model_ignores_flag_for_cache(tmp_path, monkeypatch):
    plain = run_process_file(tmp_path, monkeypatch, ["--model", "gpt-4o-transcribe-diarize"])
    flagged = run_process_file(tmp_path, monkeypatch, ["--model", "gpt-4o-transcribe-diarize", "--diarize"])
    assert plain == flagged
    assert plain[0][1] is False


# --- warn before paying when --diarize cannot keep labels consistent --------

def test_early_warning_when_diarize_needs_several_requests(tmp_path, monkeypatch):
    logs = []
    run_process_file(tmp_path, monkeypatch, ["--diarize", "--dry-run"], duration=13000.0, logs=logs)
    assert any("restart" in m and "label" in m for m in logs)


def test_no_warning_for_single_request(tmp_path, monkeypatch):
    logs = []
    run_process_file(tmp_path, monkeypatch, ["--diarize", "--dry-run"], duration=2400.0, logs=logs)
    assert not any("restart" in m for m in logs)


# --- retry loop ----------------------------------------------------------------

def fake_client(create):
    transcriptions = type("T", (), {"create": staticmethod(create)})()
    return type("C", (), {"audio": type("A", (), {"transcriptions": transcriptions})()})()


def make_chunk(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 4096)
    return t.Chunk(0, 0.0, 1.0, audio, tmp_path / "a.json")


def test_in_flight_exhaustion_on_single_chunk_does_not_suggest_workers(tmp_path, monkeypatch):
    body = {"message": "busy", "code": 402, "metadata": {"reason": "in_flight_budget_exhausted"}}

    def create(**kwargs):
        raise FakeError(402, 402, body)

    monkeypatch.setattr(t, "_pause", lambda s: None)
    monkeypatch.setattr(t, "log", lambda m: None)
    with pytest.raises(RuntimeError) as caught:
        t.transcribe_chunk(fake_client(create), make_chunk(tmp_path), MAI, "ru", None, concurrent=False)
    assert "--workers 1" not in str(caught.value)
    assert "credits" in str(caught.value)


def test_pause_returns_at_once_after_abort():
    t._ABORT.set()
    try:
        started = time.monotonic()
        t._pause(5)
        assert time.monotonic() - started < 1
    finally:
        t._ABORT.clear()


def test_diarize_400_is_not_downgraded_to_plain_json(tmp_path, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise FakeError(400, 400, {"message": "response_format verbose_json is not supported with diarization"})

    monkeypatch.setattr(t, "log", lambda m: None)
    with pytest.raises(t.FatalAPIError):
        t.transcribe_chunk(fake_client(create), make_chunk(tmp_path), MAI, "ru", None, diarize=True)
    assert len(calls) == 1


def test_non_diarize_400_still_falls_back_to_json(tmp_path, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["response_format"] != "json":
            raise FakeError(400, 400, {"message": "response_format verbose_json is not supported"})
        return type("R", (), {"model_dump": lambda self: {"text": "ok", "segments": []}})()

    monkeypatch.setattr(t, "log", lambda m: None)
    assert t.transcribe_chunk(fake_client(create), make_chunk(tmp_path), MAI, "ru", None)["text"] == "ok"
    assert [c["response_format"] for c in calls] == ["verbose_json", "json"]


# --- response parsing ----------------------------------------------------------

def test_words_do_not_override_speaker_segments():
    payload = {"segments": [{"start": 0, "end": 2, "text": "a b", "speaker": 0}],
               "words": [{"word": "a", "start": 0, "end": .5}, {"word": "b", "start": 1.5, "end": 2}]}
    assert t.payload_segments(payload, 0) == [{"start": 0.0, "end": 2.0, "text": "a b", "speaker": "0"}]


# --- texts -----------------------------------------------------------------------

def test_missing_key_under_diarize_suggests_openai_diarize_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    with pytest.raises(t.MissingKeyError) as caught:
        t.make_client("openrouter", diarize=True)
    assert "gpt-4o-transcribe-diarize" in str(caught.value)
    assert "whisper-1" not in str(caught.value)


def test_speakers_format_error_shows_numeric_example(capsys):
    with pytest.raises(SystemExit):
        t.parse_args(["--diarize", "--speakers", "nonsense"])
    assert "expects pairs like  0=" in capsys.readouterr().err


def test_chunk_help_mentions_diarize(capsys):
    with pytest.raises(SystemExit):
        t.parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "4792 s with --diarize" in help_text
