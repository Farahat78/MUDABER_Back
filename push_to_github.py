"""
push_to_github.py
────────────────────────────────────────────────────────────────────────────────
Commits and pushes the canonical data files to the GitHub repository so that
Streamlit Cloud can read the latest data.

Requirements:
  - Git must be installed and configured.
  - The pipeline repo must be a git repository.
  - SSH key or credential helper must be configured for push access.

Usage (called by run_pipeline.py --push):
    python push_to_github.py
"""

import os
import subprocess
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Files to commit (relative to repo root)
DATA_FILES = [
    "data/latest_products.csv",
    "data/predictions.csv",
]


def _run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def push_data_files(repo_dir: str):
    """
    Stage, commit, and push the canonical CSV data files.

    Args:
        repo_dir: Root directory of the prediction-pipeline git repository.
    """
    logger.info(f"Starting GitHub push from: {repo_dir}")

    # Verify it's a git repo
    result = _run_git(["status", "--short"], cwd=repo_dir, check=False)
    if result.returncode != 0:
        logger.error("Not a git repository. Run `git init` and connect to GitHub first.")
        return

    # Stage only the data files
    files_to_add = []
    for f in DATA_FILES:
        full_path = os.path.join(repo_dir, f)
        if os.path.exists(full_path):
            files_to_add.append(f)
            logger.info(f"  Staging: {f} ({os.path.getsize(full_path):,} bytes)")
        else:
            logger.warning(f"  File not found, skipping: {full_path}")

    if not files_to_add:
        logger.warning("No data files to push.")
        return

    _run_git(["add"] + files_to_add, cwd=repo_dir)

    # Check if there is anything to commit
    status = _run_git(["status", "--short"], cwd=repo_dir, check=False)
    if not status.stdout.strip():
        logger.info("Nothing to commit — data files unchanged.")
        return

    # Commit
    commit_msg = f"[pipeline] Update data — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    _run_git(["commit", "-m", commit_msg], cwd=repo_dir)
    logger.info(f"Committed: {commit_msg}")

    # Push
    _run_git(["push"], cwd=repo_dir)
    logger.info("✅ Pushed to GitHub successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    push_data_files(os.path.dirname(os.path.abspath(__file__)))
