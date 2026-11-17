"""Regressions from the final branch review."""
import httpx
from openai import OpenAI

import transcribe as t


def sdk_error(status: int, body: dict):
    """A real openai.APIStatusError, built the way the SDK builds it from a response."""
    client = OpenAI(api_key="x", base_url="https://openrouter.ai/api/v1")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions")
    response = httpx.Response(status, json=body, request=request)
    return client._make_status_error_from_response(response)


def test_wrapped_provider_400_drops_the_named_parameter():
    error = sdk_error(400, {"error": {"code": 400, "message": "Provider returned error",
                                      "metadata": {"raw": "timestamp_granularities not supported",
                                                   "provider_name": "xAI"}}})
    options = {"model": "x/y", "response_format": "verbose_json",
               "timestamp_granularities": ["word", "segment"]}
    assert t.drop_unsupported_option(options, error) == "timestamp_granularities"
    assert "timestamp_granularities" not in options


def test_api_message_includes_provider_raw_text():
    error = sdk_error(400, {"error": {"code": 400, "message": "Provider returned error",
                                      "metadata": {"raw": {"detail": "bad audio"}}}})
    assert "bad audio" in t.api_message(error)


def test_non_string_param_does_not_crash():
    error = sdk_error(400, {"error": {"code": 400, "message": "nope", "param": 123}})
    assert t.drop_unsupported_option({"model": "x/y", "response_format": "json"}, error) is None


def test_chunk_text_without_segments_becomes_one_segment():
    segs = t.payload_segments({"text": " Короткий хвост. ", "segments": []}, offset=60, end=75)
    assert segs == [{"start": 60.0, "end": 75.0, "text": "Короткий хвост."}]


def test_empty_text_without_segments_stays_empty():
    assert t.payload_segments({"text": "", "segments": []}, offset=60, end=75) == []


def test_words_without_timings_are_skipped():
    words = [{"word": "раз", "start": 0, "end": .3}, {"word": "два", "start": None, "end": None},
             {"word": "три"}, {"word": "четыре", "start": .5, "end": .9}]
    segs = t.segments_from_words(words, 0)
    assert [s["text"] for s in segs] == ["раз четыре"]
