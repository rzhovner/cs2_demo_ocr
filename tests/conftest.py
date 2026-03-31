"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `from src.X import Y` works
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GROUND_TRUTH_DIR = PROJECT_ROOT / "tests" / "ground_truth"


@pytest.fixture
def ground_truth_dir():
    """Path to the ground_truth/ directory."""
    return GROUND_TRUTH_DIR


@pytest.fixture
def training1_csv():
    """Path to training1 engagement CSV, or skip if not available."""
    path = GROUND_TRUTH_DIR / "training1_engagements.csv"
    if not path.exists():
        pytest.skip("training1_engagements.csv not committed yet")
    return path


@pytest.fixture
def training2_csv():
    """Path to training2 (spraying) engagement CSV, or skip if not available."""
    path = GROUND_TRUTH_DIR / "training2_spraying_engagements.csv"
    if not path.exists():
        pytest.skip("training2_spraying_engagements.csv not committed yet")
    return path
