"""Small runtime metadata helpers shared by forecasting workflows."""

from __future__ import annotations

import subprocess


def git_output(*args: str, required: bool = False) -> str | None:
    """Run a Git command and optionally fail when provenance cannot be recorded."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        if required:
            raise RuntimeError("Git is required to record run provenance.") from error
        return None
    except subprocess.CalledProcessError as error:
        if required:
            stderr = (error.stderr or "").strip()
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"Could not run git {' '.join(args)}{detail}") from error
        return None

    output = result.stdout.strip()
    if required and not output:
        raise RuntimeError(f"git {' '.join(args)} returned no output.")
    return output


def git_commit() -> str:
    """Return the exact checked-out commit or fail clearly."""
    commit = git_output("rev-parse", "HEAD", required=True)
    assert commit is not None
    return commit


def git_is_dirty() -> bool:
    """Return whether tracked or untracked files differ from HEAD."""
    return bool(git_output("status", "--porcelain"))
