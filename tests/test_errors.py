import pytest

import transcribe as t

MAI = "microsoft/mai-transcribe-2"


class FakeError(Exception):
    """Shaped like openai.APIStatusError: status_code, code, body, response."""

    def __init__(self, status, code=None, body=None, headers=None):
        super().__init__((body or {}).get("message", "error"))
        self.status_code = status
        self.code = code
        self.body = body
        self.response = type("R", (), {"headers": headers or {}})()


def test_in_flight_402_is_retryable():
    body = {"message": "This request would exceed your available credits given your "
                       "current in-flight requests.", "code": 402,
            "metadata": {"reason": "in_flight_budget_exhausted",
                         "headers": {"Retry-After": "120"}}}
    error = FakeError(402, 402, body)
    assert t.fatal_reason(error, MAI) is None
    assert t.retry_after_seconds(error) == 120


def test_other_402_is_fatal_with_credits_link():
    error = FakeError(402, 402, {"message": "Insufficient credits", "code": 402})
    assert "openrouter.ai/settings/credits" in t.fatal_reason(error, MAI)


def test_integer_code_does_not_crash():
    assert t.fatal_reason(FakeError(500, 500, {"code": 500}), MAI) is None


def test_401_names_the_right_key():
    assert "OPENROUTER_API_KEY" in t.fatal_reason(FakeError(401, 401), MAI)
    assert "OPENAI_API_KEY" in t.fatal_reason(FakeError(401, "invalid_api_key"), "whisper-1")


def test_openai_insufficient_quota_still_fatal():
    assert "out of credit" in t.fatal_reason(FakeError(429, "insufficient_quota"), "whisper-1")


def test_header_retry_after_wins():
    error = FakeError(429, None, {"message": "slow down"}, headers={"retry-after": "7"})
    assert t.retry_after_seconds(error) == 7


def test_402_text_names_openrouter_not_model():
    msg = t.fatal_reason(FakeError(402, 402, {"message": "Insufficient credits", "code": 402}), MAI)
    assert "OpenRouter" in msg
    assert "microsoft" not in msg


def test_403_includes_api_message():
    msg = t.fatal_reason(FakeError(403, 403, {"message": "flagged by moderation"}), MAI)
    assert "flagged by moderation" in msg


def test_in_flight_402_detector():
    body = {"message": "x", "code": 402, "metadata": {"reason": "in_flight_budget_exhausted"}}
    assert t.is_in_flight_wait(FakeError(402, 402, body))
    assert not t.is_in_flight_wait(FakeError(402, 402, {"message": "x"}))
    assert not t.is_in_flight_wait(FakeError(500, 500, body))


def test_in_flight_waits_do_not_burn_attempts(monkeypatch, tmp_path):
    """Five in-flight 402s, then success: more than MAX_ATTEMPTS waits must not fail the chunk."""
    body = {"message": "busy", "code": 402, "metadata": {"reason": "in_flight_budget_exhausted"}}
    calls = {"n": 0}

    class Response:
        def model_dump(self):
            return {"text": "ок", "segments": [{"start": 0, "end": 1, "text": "ок"}]}

    class Transcriptions:
        def create(self, **kwargs):
            calls["n"] += 1
            if calls["n"] <= t.MAX_ATTEMPTS + 1:
                raise FakeError(402, 402, body)
            return Response()

    client = type("C", (), {"audio": type("A", (), {"transcriptions": Transcriptions()})()})()
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 4096)
    chunk = t.Chunk(0, 0.0, 1.0, audio, tmp_path / "a.json")
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    monkeypatch.setattr(t, "log", lambda m: None)
    assert t.transcribe_chunk(client, chunk, MAI, "ru", None)["text"] == "ок"


def test_too_many_in_flight_waits_suggest_one_worker(monkeypatch, tmp_path):
    body = {"message": "busy", "code": 402, "metadata": {"reason": "in_flight_budget_exhausted"}}

    class Transcriptions:
        def create(self, **kwargs):
            raise FakeError(402, 402, body)

    client = type("C", (), {"audio": type("A", (), {"transcriptions": Transcriptions()})()})()
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x" * 4096)
    chunk = t.Chunk(0, 0.0, 1.0, audio, tmp_path / "a.json")
    monkeypatch.setattr(t.time, "sleep", lambda s: None)
    monkeypatch.setattr(t, "log", lambda m: None)
    with pytest.raises(RuntimeError, match="--workers 1"):
        t.transcribe_chunk(client, chunk, MAI, "ru", None)
