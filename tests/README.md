# Tests

## Testing Approach

This project uses **ground truth validation** rather than traditional unit tests. The demo parser (`cs2_engagement_parser.py`) produces canonical engagement CSVs from `.dem` files. All video/audio detectors validate against these CSVs.

## Ground Truth

The `ground_truth/` directory contains committed engagement CSVs. These are small text files generated from `.dem` files via the demo parser. Since `.dem` files are large binaries (gitignored), the CSVs are committed so tests can run without the original demos.

To regenerate ground truth from a new `.dem` file:
```bash
python src/cs2_engagement_parser.py data/your_demo.dem --output tests/ground_truth/your_demo_engagements.csv
```

## Running Tests

```bash
# Install test dependencies
pip install pytest scipy

# Run all tests
python -m pytest tests/ -v

# Run only scoring unit tests (no external data needed)
python -m pytest tests/test_scoring.py -v

# Run detector integration tests (requires ground truth CSVs + media files)
python -m pytest tests/test_detectors.py -v
```

## Test Structure

| File | Type | Dependencies |
|---|---|---|
| `test_scoring.py` | Unit tests | None (synthetic data) |
| `test_detectors.py` | Integration tests | Ground truth CSVs + video/audio files in `data/` |
| `conftest.py` | Fixtures | Loads ground truth CSVs |
