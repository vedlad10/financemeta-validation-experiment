# FinanceMeta: random split vs purged walk-forward

This repository is the smallest reproducible experiment requested by Ryan Gomez. It predicts one equity's next-day return with one frozen feature set and one Ridge model family, changing only the validation protocol.

## Run

```powershell
python experiment.py
```

Protocol-only test (does not rerun the empirical lockbox):

```powershell
python -m unittest -v test_protocol.py
```

The script downloads daily AAPL prices from Yahoo's chart endpoint, adjusts OHLC by the supplied adjusted-close factor, caches the raw JSON, verifies all split invariants, and writes:

- `results/validation_scores.csv`
- `results/fold_boundaries.csv`
- `results/lockbox_predictions.csv`
- `results/summary.json`
- `results/comparison.png`

To verify the machinery without network access:

```powershell
python experiment.py --synthetic-smoke-test
```

Synthetic output is clearly labeled and is not evidence about the empirical hypothesis.

The protocol is frozen in [SPEC.md](SPEC.md); related primary sources are in [PAPERS.md](PAPERS.md). Do not modify `config.json` after reading final lockbox results. Every run records its config hash.

The preserved empirical outcome is summarized in [RESULTS.md](RESULTS.md), and [RYAN_UPDATE.md](RYAN_UPDATE.md) is a concise ready-to-send note.
