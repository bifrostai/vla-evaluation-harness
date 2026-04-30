"""Host-side cache directory resolver and runtime helpers for vla-eval.

Single source of truth for "where does this thing go on the host" and
the runtime hooks that consumers (benchmarks, model servers) use to
acquire externally-licensed assets.

Layout (mirrors HuggingFace's ``HF_HOME`` / ``HF_ASSETS_CACHE``):

    ${VLA_EVAL_HOME, $XDG_CACHE_HOME/vla-eval, ~/.cache/vla-eval}/
        assets/                     # ``$VLA_EVAL_ASSETS_CACHE`` overrides
            behavior1k/             # OmniGibson scenes + task instances
            vlanext/                # github.com/DravenALG/VLANeXt clone
            mme-vla/                # github.com/RoboMME/robomme_policy_learning clone
            ...

The terminology follows HF's: "assets" are workflow-related files that
the harness (or a model-server library) downloads from somewhere other
than HF Hub itself — scene meshes, helper repos, per-task JSONs.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def home() -> Path:
    """vla-eval root directory.

    Precedence: ``$VLA_EVAL_HOME > $XDG_CACHE_HOME/vla-eval > ~/.cache/vla-eval``.
    """
    override = os.environ.get("VLA_EVAL_HOME")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "vla-eval"


def assets_cache(subdir: str | None = None) -> Path:
    """Per-component assets directory.

    Precedence: ``$VLA_EVAL_ASSETS_CACHE > home()/assets``, with
    optional ``subdir`` appended.
    """
    override = os.environ.get("VLA_EVAL_ASSETS_CACHE")
    base = Path(override).expanduser() if override else home() / "assets"
    return base / subdir if subdir else base


def ensure_git_clone(name: str, repo: str, rev: str, *, shallow: bool = False) -> Path:
    """Lazy clone ``repo`` at ``rev`` into ``assets_cache(name)``.

    Idempotent — returns immediately when ``<target>/.git`` already exists.
    ``shallow=True`` does ``--depth 1 --branch rev`` (branch / tag only).
    ``shallow=False`` does a full clone followed by ``git checkout rev``
    (works for arbitrary commit SHAs).
    """
    target = assets_cache(name)
    if (target / ".git").exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s @ %s -> %s", repo, rev, target)
    if shallow:
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", rev, repo, str(target)])
    else:
        subprocess.check_call(["git", "clone", repo, str(target)])
        subprocess.check_call(["git", "-C", str(target), "checkout", rev])
    return target


def ensure_license(license_id: str, *, url: str, description: str) -> None:
    """Ensure the user has accepted ``license_id``; raise on rejection.

    Acceptance order:
        1. ``$VLA_EVAL_ACCEPTED_LICENSES`` env var (comma-separated);
           membership of ``license_id`` counts as accepted.
        2. Interactive stdin (``isatty``): prints ``description`` plus
           ``url`` to stderr and reads ``y/N``.
        3. Otherwise: ``SystemExit`` with a hint pointing at
           ``--accept-license <id>`` / ``VLA_EVAL_ACCEPTED_LICENSES=<id>``.
    """
    accepted = {item.strip() for item in os.environ.get("VLA_EVAL_ACCEPTED_LICENSES", "").split(",") if item.strip()}
    if license_id in accepted:
        return

    msg = f"Licence required: {description}\n  ID:  {license_id}\n  URL: {url}\n"

    if not sys.stdin.isatty():
        sys.stderr.write(msg)
        sys.stderr.write(
            f"\nNon-interactive context (no TTY).  Re-run with "
            f"--accept-license {license_id} or set "
            f"VLA_EVAL_ACCEPTED_LICENSES={license_id} in the environment.\n"
        )
        raise SystemExit(1)

    sys.stderr.write(msg)
    sys.stderr.write("Accept this licence? [y/N] ")
    sys.stderr.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer in ("y", "yes"):
        return
    sys.stderr.write("Licence rejected; aborting.\n")
    raise SystemExit(1)
