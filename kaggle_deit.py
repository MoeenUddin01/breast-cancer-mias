"""Kaggle notebook script for DeiT transformer training on BreakHis.

Run in Kaggle with:
    exec(open("kaggle_deit.py").read(), globals())
"""

from __future__ import annotations

import atexit
import fcntl
import os
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/MoeenUddin01/breast-cancer-mias.git"
KAGGLE_REPO_PATH = Path("/kaggle/working/breast-cancer-mias")
LOCK_PATH = Path("/tmp/deit_training.lock")


def _resolve_repo_path() -> Path:
    """Resolve repo path in notebook/script execution modes."""
    if KAGGLE_REPO_PATH.exists():
        return KAGGLE_REPO_PATH
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def _ensure_repo_ready(repo_path: Path) -> Path:
    """Clone repo in Kaggle if missing, then set cwd and sys.path."""
    if str(repo_path).startswith("/kaggle/") and not repo_path.exists():
        print(f"⏳ Cloning repository to {repo_path} ...")
        subprocess.run(["git", "clone", REPO_URL, str(repo_path)], check=True)

    os.chdir(str(repo_path))
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    print(f"✅ Repo ready at: {repo_path}")
    return repo_path


def _acquire_single_run_lock() -> object | None:
    """Acquire a non-blocking lock so only one run executes."""
    lock_file = open(LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None

    lock_file.write(str(os.getpid()))
    lock_file.flush()

    def _cleanup_lock() -> None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            if LOCK_PATH.exists():
                LOCK_PATH.unlink()
        except OSError:
            pass

    atexit.register(_cleanup_lock)
    return lock_file


def main() -> None:
    """Run full DeiT training pipeline."""
    lock_file = _acquire_single_run_lock()
    if lock_file is None:
        print("⚠️ DeiT training is already running. Stop other run and retry.")
        return

    repo_path = _ensure_repo_ready(_resolve_repo_path())
    train_script = repo_path / "kaggle_deit_train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing training script: {train_script}")

    print("🚀 Starting DeiT training script...")
    subprocess.run([sys.executable, str(train_script)], check=True)
    _ = lock_file  # Keep handle alive for full process lifetime.


main()
