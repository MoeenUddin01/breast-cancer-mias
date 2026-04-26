"""Kaggle entrypoint for DeiT training.

Use this file in notebooks:
    exec(open("kaggle_deit.py").read(), globals())
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Execute the existing DeiT training script."""
    script_path = Path(__file__).resolve().parent / "kaggle_deit_train.py"
    exec(script_path.read_text(encoding="utf-8"), globals())


if __name__ == "__main__":
    main()
