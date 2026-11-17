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
    spec = t.MODEL_SPECS[args.model]
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
    assert t.MODEL_SPECS[args.model].max_seconds == 1400


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
