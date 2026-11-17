import pytest

import transcribe as t


def test_aliases_resolve_to_full_ids():
    assert t.resolve_model("mai") == "microsoft/mai-transcribe-2"
    assert t.resolve_model("voxtral") == "mistralai/voxtral-small-transcribe-3-0"
    assert t.resolve_model("scribe") == "elevenlabs/scribe-v2"
    assert t.resolve_model("whisper-1") == "whisper-1"


def test_winners_are_openrouter_with_timestamps():
    for alias in ("mai", "voxtral", "scribe"):
        spec = t.MODEL_SPECS[t.resolve_model(alias)]
        assert spec.provider == "openrouter"
        assert spec.timestamps
        assert not spec.supports_prompt
    assert t.MODEL_SPECS["elevenlabs/scribe-v2"].word_timestamps
    assert t.MODEL_SPECS["whisper-1"].provider == "openai"
    assert t.DEFAULT_MODEL == "microsoft/mai-transcribe-2"


def test_winner_prices_per_minute():
    assert t.MODEL_SPECS["microsoft/mai-transcribe-2"].price_per_minute == pytest.approx(0.00167)
    assert t.MODEL_SPECS["mistralai/voxtral-small-transcribe-3-0"].price_per_minute == pytest.approx(0.004)
    assert t.MODEL_SPECS["elevenlabs/scribe-v2"].price_per_minute == pytest.approx(0.00367)


def test_unknown_openrouter_id_is_registered_with_catalog_price():
    model = t.resolve_model("x-ai/grok-stt-2.0",
                            catalog_loader=lambda: {"x-ai/grok-stt-2.0": 0.00167})
    spec = t.MODEL_SPECS[model]
    assert model == "x-ai/grok-stt-2.0"
    assert spec.provider == "openrouter"
    assert spec.price_per_minute == 0.00167
    assert spec.response_format == "verbose_json"


def test_typo_in_openrouter_id_fails_with_suggestion():
    with pytest.raises(ValueError, match="microsoft/mai-transcribe-2"):
        t.resolve_model("microsoft/mai-transcribe-3",
                        catalog_loader=lambda: {"microsoft/mai-transcribe-2": 0.00167})


def test_catalog_offline_still_accepts_id():
    def offline():
        raise OSError("no network")
    model = t.resolve_model("acme/new-stt", catalog_loader=offline)
    assert t.MODEL_SPECS[model].provider == "openrouter"
    assert t.MODEL_SPECS[model].price_per_minute is None


def test_known_model_skips_catalog():
    def explode():
        raise AssertionError("catalog must not be fetched for built-in models")
    assert t.resolve_model("microsoft/mai-transcribe-2", catalog_loader=explode)


def test_unknown_plain_name_fails():
    with pytest.raises(ValueError, match="--list-models"):
        t.resolve_model("whisper-9")


def test_catalog_price_parsing():
    data = {"data": [
        {"id": "a/sec", "pricing": {"prompt": "0.0000277777777778"}},
        {"id": "b/tok", "pricing": {"prompt": "0.000002", "completion": "0.000012"}},
    ]}
    prices = t.catalog_prices(data)
    assert prices["a/sec"] == pytest.approx(0.00166667, rel=1e-3)
    assert prices["b/tok"] is None
