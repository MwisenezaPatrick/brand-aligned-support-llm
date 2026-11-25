from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = [
    "app/api",
    "app/frontend",
    "app/inference",
    "app/integrations",
    "artifacts",
    "configs",
    "data/raw",
    "data/processed",
    "docs",
    "evaluation/reports",
    "scripts",
    "tests",
    "training",
]

REQUIRED_FILES = [
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "requirements-lock.txt",
]


def test_required_directories_exist():
    missing = [d for d in REQUIRED_DIRECTORIES if not (ROOT / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_required_files_exist():
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    assert not missing, f"Missing files: {missing}"