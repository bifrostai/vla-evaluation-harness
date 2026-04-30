"""Tests for the host cache resolver and ``ensure_license`` helper."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from vla_eval import dirs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip cache-related env vars so each test starts from defaults."""
    for var in ("VLA_EVAL_HOME", "VLA_EVAL_ASSETS_CACHE", "VLA_EVAL_ACCEPTED_LICENSES", "XDG_CACHE_HOME"):
        monkeypatch.delenv(var, raising=False)


def test_home_default() -> None:
    assert dirs.home() == Path.home() / ".cache" / "vla-eval"


def test_home_xdg_cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert dirs.home() == tmp_path / "vla-eval"


def test_home_vla_eval_home_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLA_EVAL_HOME", str(tmp_path / "root"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "ignored"))
    assert dirs.home() == tmp_path / "root"


def test_assets_cache_default() -> None:
    assert dirs.assets_cache() == Path.home() / ".cache" / "vla-eval" / "assets"
    assert dirs.assets_cache("foo") == Path.home() / ".cache" / "vla-eval" / "assets" / "foo"


def test_assets_cache_subdir_invariant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLA_EVAL_HOME", str(tmp_path))
    assert dirs.assets_cache("foo") == dirs.assets_cache() / "foo"


def test_assets_cache_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLA_EVAL_HOME", str(tmp_path / "ignored"))
    monkeypatch.setenv("VLA_EVAL_ASSETS_CACHE", str(tmp_path / "fast-ssd"))
    assert dirs.assets_cache("foo") == tmp_path / "fast-ssd" / "foo"


def test_ensure_license_env_var_bypasses_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLA_EVAL_ACCEPTED_LICENSES", "alpha,behavior-dataset-tos,beta")
    dirs.ensure_license("behavior-dataset-tos", url="https://x", description="y")  # no raise


def test_ensure_license_interactive_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    dirs.ensure_license("any", url="https://x", description="y")  # no raise


def test_ensure_license_interactive_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    with pytest.raises(SystemExit):
        dirs.ensure_license("any", url="https://x", description="y")


def test_ensure_license_non_tty_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit):
        dirs.ensure_license("any", url="https://x", description="y")
