"""Integration tests for kill detectors.

These tests run detectors against committed ground truth CSVs and assert
minimum accuracy thresholds. They require:
  - Ground truth CSVs in tests/ground_truth/
  - Video/audio files in data/ (for video-based detectors)

Most tests are skipped by default until the required data files are available.
"""

import pytest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _video_path(name):
    """Return path to a video file in data/, or skip if missing."""
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not found in data/")
    return str(path)


class TestAudioKillDetector:
    """Audio detector should achieve documented accuracy thresholds."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from src.audio_kill_detector import extract_audio, compute_energy, detect_kills
            self.extract_audio = extract_audio
            self.compute_energy = compute_energy
            self.detect_kills = detect_kills
        except ImportError:
            pytest.skip("audio_kill_detector dependencies not available")

    @pytest.mark.skip(reason="Requires video file in data/")
    def test_tapping_recall(self, training1_csv):
        """Tapping session should achieve >= 95% recall."""
        # When data is available, uncomment and configure:
        # video = _video_path("training1_03172026.mp4")
        # audio, sr = self.extract_audio(video)
        # energy, times = self.compute_energy(audio, sr)
        # kills, _, _ = self.detect_kills(energy, times)
        # ... validate against training1_csv ...
        pass

    @pytest.mark.skip(reason="Requires video file in data/")
    def test_spraying_recall(self, training2_csv):
        """Spraying session should achieve >= 88% recall."""
        pass


class TestDemoParser:
    """Demo parser should produce well-formed engagement CSVs."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.cs2_engagement_parser import build_engagement_map
        self.build_engagement_map = build_engagement_map

    @pytest.mark.skip(reason="Requires .dem file in data/")
    def test_engagement_count(self):
        """A 100-bot challenge should produce ~100 engagements."""
        pass

    def test_import_succeeds(self):
        """Verify the parser module can be imported."""
        from src import cs2_engagement_parser
        assert hasattr(cs2_engagement_parser, "main")
