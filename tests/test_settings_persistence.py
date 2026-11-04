import json
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from app.core import config


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", str(path))
    return path


def test_saved_settings_replace_the_previous_document(settings_path):
    settings_path.write_text('{"provider": "local", "api_key": "previous-value"}')
    settings = config.LLMSettings(
        provider="custom", model="local-model", api_key="new-value"
    )

    config.save_settings(settings)

    assert config.load_settings() == settings
    assert json.loads(settings_path.read_text())["api_key"] == "new-value"
    assert list(settings_path.parent.glob(".settings-*.tmp")) == []


def test_serialization_failure_preserves_previous_settings(settings_path):
    previous = '{"provider": "local", "api_key": "previous-value"}'
    settings_path.write_text(previous)

    with patch.object(
        config.LLMSettings, "model_dump_json", side_effect=ValueError("serialize")
    ):
        with pytest.raises(ValueError, match="serialize"):
            config.save_settings(config.LLMSettings())

    assert settings_path.read_text() == previous
    assert list(settings_path.parent.glob(".settings-*.tmp")) == []


def test_replace_failure_preserves_previous_settings(settings_path):
    previous = '{"provider": "local", "api_key": "previous-value"}'
    settings_path.write_text(previous)

    with patch("os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            config.save_settings(config.LLMSettings(api_key="new-value"))

    assert settings_path.read_text() == previous
    assert list(settings_path.parent.glob(".settings-*.tmp")) == []


def test_partial_write_failure_preserves_previous_settings(settings_path):
    previous = '{"provider": "local", "api_key": "previous-value"}'
    settings_path.write_text(previous)

    @contextmanager
    def failing_temporary_file(*args, **kwargs):
        with NamedTemporaryFile(*args, **kwargs) as temporary:

            class PartialWriter:
                name = temporary.name

                def write(self, payload):
                    temporary.write(payload[:5])
                    temporary.flush()
                    raise OSError("write failed")

            yield PartialWriter()

    with patch("tempfile.NamedTemporaryFile", failing_temporary_file):
        with pytest.raises(OSError, match="write failed"):
            config.save_settings(config.LLMSettings(api_key="new-value"))

    assert settings_path.read_text() == previous
    assert list(settings_path.parent.glob(".settings-*.tmp")) == []


def test_first_save_creates_a_complete_document(settings_path):
    settings = config.LLMSettings(model="local-model")
    config.save_settings(settings)
    assert config.load_settings() == settings


def test_failed_first_serialization_does_not_create_an_empty_file(settings_path):
    with patch.object(
        config.LLMSettings, "model_dump_json", side_effect=ValueError("serialize")
    ):
        with pytest.raises(ValueError, match="serialize"):
            config.save_settings(config.LLMSettings())
    assert not settings_path.exists()


def test_unicode_settings_round_trip(settings_path):
    settings = config.LLMSettings(model="نموذج محلي")
    config.save_settings(settings)
    assert config.load_settings() == settings
