"""Kaggle entrypoint for DeiT training.

Use this file in notebooks:
    exec(open("kaggle_deit.py").read(), globals())
"""

from __future__ import annotations

from pathlib import Path


def _resolve_repo_root() -> Path:
    """Resolve repository root in script and notebook contexts."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def main() -> None:
    """Execute the existing DeiT training script."""
    script_path = _resolve_repo_root() / "kaggle_deit_train.py"
    exec(script_path.read_text(encoding="utf-8"), globals())


if __name__ == "__main__":
    main()
