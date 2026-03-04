"""Tests for settings parsing behavior."""
from app.core.config import Settings


def test_debug_release_string_parses_to_false():
    settings = Settings(DEBUG="release")
    assert settings.DEBUG is False


def test_debug_dev_string_parses_to_true():
    settings = Settings(DEBUG="development")
    assert settings.DEBUG is True
