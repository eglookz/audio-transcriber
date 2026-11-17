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
