import pytest

import transcribe as t


def test_parse_args_resolves_alias():
    assert t.parse_args(["--model", "voxtral"]).model == "mistralai/voxtral-small-transcribe-3-0"


def test_parse_args_default_is_mai():
    assert t.parse_args([]).model == "microsoft/mai-transcribe-2"


def test_parse_args_rejects_unknown_model():
    with pytest.raises(SystemExit):
        t.parse_args(["--model", "whisper-9"])


def test_missing_openrouter_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    with pytest.raises(t.MissingKeyError, match="OPENROUTER_API_KEY"):
        t.make_client("openrouter")


def test_only_openai_key_with_default_model_points_to_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    with pytest.raises(t.MissingKeyError, match="openrouter.ai/settings/keys"):
        t.make_client(t.MODEL_SPECS[t.DEFAULT_MODEL].provider)


def test_openrouter_client_points_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    assert str(t.make_client("openrouter").base_url).startswith("https://openrouter.ai/api/v1")


def test_openai_client_uses_default_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    assert "api.openai.com" in str(t.make_client("openai").base_url)


def test_list_models_shows_aliases_and_providers(capsys):
    t.print_model_table()
    out = capsys.readouterr().out
    assert "mai" in out and "openrouter" in out and "whisper-1" in out
    assert "openrouter.ai/collections/speech-to-text-models" in out


def test_missing_openrouter_key_suggests_whisper_when_openai_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(t, "load_dotenv", lambda path: None)
    with pytest.raises(t.MissingKeyError, match="--model whisper-1"):
        t.make_client("openrouter")


def test_foreign_cache_count(tmp_path):
    (tmp_path / "000_000000000-000060000_aaaaaaaa.json").write_text("{}")
    (tmp_path / "001_000060000-000120000_aaaaaaaa.json").write_text("{}")
    (tmp_path / "000_000000000-000060000_bbbbbbbb.json").write_text("{}")
    (tmp_path / "000_000000000-000060000_16000_48k.mp3").write_bytes(b"")
    assert t.foreign_cache_count(tmp_path, "aaaaaaaa") == 1
    assert t.foreign_cache_count(tmp_path / "missing", "aaaaaaaa") == 0
