import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcribe  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_model_specs():
    """resolve_model and --diarize register specs globally; undo that after each test."""
    saved = dict(transcribe.MODEL_SPECS)
    yield
    transcribe.MODEL_SPECS.clear()
    transcribe.MODEL_SPECS.update(saved)
