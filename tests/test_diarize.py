import json

import pytest

import transcribe as t

MAI = "microsoft/mai-transcribe-2"


def test_mai_has_diarize_options():
    assert t.MODEL_SPECS[MAI].diarize_options == {"azure": {"diarization": {"enabled": True}}}


def test_extra_body_is_json_string():
    body = t.diarize_extra_body({"azure": {"diarization": {"enabled": True}}})
    assert isinstance(body["provider"], str)
    assert json.loads(body["provider"]) == {"options": {"azure": {"diarization": {"enabled": True}}}}


def test_diarize_flag_switches_spec():
    args = t.parse_args(["--diarize"])
    spec = args.spec
    assert args.diarize
    assert spec.diarizes
    assert spec.default_chunk_seconds == 4800
    assert spec.max_seconds == 6000
    assert args.chunk <= int((6000 - t.MODEL_LIMIT_MARGIN) / t.CHUNK_MAX_FACTOR)


def test_diarize_rejected_for_unsupported_model(capsys):
    with pytest.raises(SystemExit):
        t.parse_args(["--model", "voxtral", "--diarize"])
    assert "mai" in capsys.readouterr().err


def test_diarize_is_noop_for_openai_diarize_model():
    args = t.parse_args(["--model", "gpt-4o-transcribe-diarize", "--diarize"])
    assert args.spec.max_seconds == 1400


def test_speakers_without_diarize_suggests_flag(capsys):
    with pytest.raises(SystemExit):
        t.parse_args(["--speakers", "0=A"])
    assert "--diarize" in capsys.readouterr().err


def test_speakers_with_numeric_labels():
    args = t.parse_args(["--diarize", "--speakers", "0=Мама,1=Консультант"])
    assert args.speakers == {"0": "Мама", "1": "Консультант"}


def test_signature_depends_on_diarize():
    assert t.request_signature("m", "ru", None) != t.request_signature("m", "ru", None, diarize=True)


def test_speaker_zero_kept():
    segs = t.payload_segments({"segments": [{"start": 0, "end": 1, "text": "a", "speaker": 0},
                                            {"start": 1, "end": 2, "text": "b", "speaker": 1}]}, 0)
    assert [s["speaker"] for s in segs] == ["0", "1"]


def test_srt_shows_speaker_zero():
    assert "0: a" in t.render_srt([{"start": 0, "end": 1, "text": "a", "speaker": "0"}])


class RecordingTranscriptions:
    def __init__(self):
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)

        class Response:
            def model_dump(self):
                return {"text": "a b", "segments": [
                    {"start": 0, "end": 1, "text": "a", "speaker": 0},
                    {"start": 1, "end": 2, "text": "b", "speaker": 1}]}
        return Response()


def fake_client(recorder):
    return type("C", (), {"audio": type("A", (), {"transcriptions": recorder})()})()


def make_chunk(tmp_path, name="a"):
    audio = tmp_path / f"{name}.mp3"
    audio.write_bytes(b"x" * 4096)
    return t.Chunk(0, 0.0, 2.0, audio, tmp_path / f"{name}.json")


def test_transcribe_chunk_sends_diarize_options_only_when_asked(tmp_path):
    recorder = RecordingTranscriptions()
    t.transcribe_chunk(fake_client(recorder), make_chunk(tmp_path, "on"), MAI, "ru", None, diarize=True)
    t.transcribe_chunk(fake_client(recorder), make_chunk(tmp_path, "off"), MAI, "ru", None)
    on, off = recorder.kwargs
    assert json.loads(on["extra_body"]["provider"])["options"]["azure"]["diarization"]["enabled"] is True
    assert "extra_body" not in off


def test_named_speaker_zero_reaches_all_outputs(tmp_path):
    segments = [{"start": 0, "end": 1, "text": "a", "speaker": "0"},
                {"start": 1, "end": 2, "text": "b", "speaker": "1"}]
    t.rename_speakers(segments, {"0": "Мама"})
    t.write_outputs(tmp_path, "x", [{"index": 0, "text": "a b", "segments": segments}], True)
    assert (tmp_path / "x.txt").read_text(encoding="utf-8-sig").startswith("Мама: a")
    assert "Мама: a" in (tmp_path / "x.srt").read_text(encoding="utf-8")
    assert "] Мама: a" in (tmp_path / "x.timestamped.txt").read_text(encoding="utf-8-sig")


def test_process_file_passes_diarize_to_request_and_cache(tmp_path, monkeypatch):
    """--diarize must reach both the API call and the cache name, end to end."""
    audio = tmp_path / "talk.mp3"
    audio.write_bytes(b"x" * 4096)
    calls = []
    monkeypatch.setattr(t, "probe_duration", lambda path: 60.0)
    monkeypatch.setattr(t, "detect_silences", lambda path, duration, show: [])
    monkeypatch.setattr(t, "extract_chunk", lambda src, dst, s, e, bitrate="48k": dst.write_bytes(b"x" * 4096))
    monkeypatch.setattr(t, "log", lambda message: None)

    def fake_transcribe(client, chunk, model, language, prompt, diarize=False, concurrent=True):
        calls.append((chunk.cache_path.name, diarize))
        return {"index": chunk.index, "start": chunk.start, "end": chunk.end,
                "text": "a", "segments": [{"start": 0, "end": 1, "text": "a"}]}

    monkeypatch.setattr(t, "transcribe_chunk", fake_transcribe)
    common = [str(audio), "--output-dir", str(tmp_path / "out"), "--work-dir", str(tmp_path / "w"),
              "--no-progress", "--force"]
    t.process_file(audio, None, t.parse_args(common))
    t.process_file(audio, None, t.parse_args(common + ["--diarize"]))
    (plain_name, plain_flag), (diar_name, diar_flag) = calls
    assert (plain_flag, diar_flag) == (False, True)
    assert plain_name != diar_name
