"""Enumerate the experiment-2 fold windows and audit their boundaries.

This script is deliberately NOT result-bearing. It fits no model, computes no
validation or lockbox score, and reads no 2026 data. It answers one question:
given the frozen calendar, exactly which rows does each protocol train on, and
do the declared boundary rules actually hold?

It imports `make_dataset` from the frozen experiment-1 module so the features
and the row index are identical by construction rather than by restatement.

    python experiment_2/plan_folds.py            writes experiment_2/fold_plan.csv

Outputs
    fold_plan.csv        one row per (protocol, fold): window dates and sizes
    fold_plan_audit.txt  the invariant checks, pass or fail
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import make_dataset  # noqa: E402  (feature code, no fitting)

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config_experiment_2.json").read_text(encoding="utf-8"))


def load_cached_frame() -> pd.DataFrame:
    """The experiment-1 cache, which covers the whole development span."""
    payload = json.loads((ROOT / CONFIG["cached_raw_path"]).read_text(encoding="utf-8"))
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    frame = pd.DataFrame({
        "Date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize(),
        "Open": quote["open"], "High": quote["high"], "Low": quote["low"],
        "Close": quote["close"], "Volume": quote["volume"], "AdjClose": adjclose,
    }).dropna()
    # Provider rule from the preregistration: adjusted close is Close, and OHL
    # are scaled by the same factor; volume is left unscaled.
    factor = frame["AdjClose"] / frame["Close"]
    for column in ["Open", "High", "Low"]:
        frame[column] = frame[column] * factor
    frame["Close"] = frame["AdjClose"]
    return frame.drop(columns="AdjClose")


def supervised_rows() -> pd.DataFrame:
    """One row per feature date, with the date its label becomes known.

    The label date comes from the raw trading calendar, not from the next row of
    the feature table. The last feature row has no successor inside the feature
    table but its label does exist on the following trading day, and taking the
    successor row instead silently drops it.
    """
    raw = load_cached_frame()
    calendar = pd.DatetimeIndex(sorted(raw["Date"].unique()))
    dataset = make_dataset(raw, CONFIG)
    positions = calendar.get_indexer(dataset.index)
    if (positions < 0).any():
        raise SystemExit("a feature date is missing from the trading calendar")
    following = positions + 1
    rows = pd.DataFrame({
        "feature_date": dataset.index,
        "label_date": [calendar[i] if i < len(calendar) else pd.NaT for i in following],
    })
    return rows.dropna(subset=["label_date"]).reset_index(drop=True)


def development(rows: pd.DataFrame) -> pd.DataFrame:
    """Membership is by label_date, never by feature date."""
    lo, hi = pd.Timestamp(CONFIG["development_label_start"]), pd.Timestamp(CONFIG["development_label_end"])
    return rows[(rows.label_date >= lo) & (rows.label_date <= hi)].reset_index(drop=True)


def validation_blocks(dev: pd.DataFrame) -> list[np.ndarray]:
    sizes = CONFIG["validation_block_sizes"]
    start = len(dev) - sum(sizes)
    if start < 0:
        raise SystemExit(f"development sample has {len(dev)} rows, too few for {sum(sizes)} validation rows")
    blocks, cursor = [], start
    for size in sizes:
        blocks.append(np.arange(cursor, cursor + size))
        cursor += size
    return blocks


def ordered_window(dev: pd.DataFrame, block: np.ndarray) -> np.ndarray:
    """Rolling origin: the most recent eligible rows strictly before the block.

    Take `train_rows + purge` candidates ending just before the block, then drop
    the purge rows closest to it, because their labels land on or after the first
    validation feature date.
    """
    n_train, purge = CONFIG["train_rows_per_fold"], CONFIG["purge_rows_before_block"]
    stop = int(block[0]) - purge
    start = stop - n_train
    if start < 0:
        raise SystemExit(f"fold starting at row {block[0]} cannot supply {n_train} earlier training rows")
    return np.arange(start, stop)


def random_window(dev: pd.DataFrame, block: np.ndarray, fold: int) -> np.ndarray:
    """The contrast arm: same size, drawn from both sides of the block."""
    n_train = CONFIG["train_rows_per_fold"]
    purge, embargo = CONFIG["purge_rows_before_block"], CONFIG["embargo_rows_after_block"]
    excluded = set(block.tolist())
    excluded.update(range(int(block[0]) - purge, int(block[0])))
    excluded.update(range(int(block[-1]) + 1, int(block[-1]) + 1 + embargo))
    candidates = np.array([i for i in range(len(dev)) if i not in excluded], dtype=int)
    if candidates.size < n_train:
        raise SystemExit(f"fold {fold}: {candidates.size} candidates, need {n_train}")
    rng = np.random.default_rng(np.random.SeedSequence([CONFIG["base_seed"], fold]))
    chosen = rng.choice(candidates, size=n_train, replace=False)
    return np.sort(chosen)          # sorted for storage only, not chronological


def describe(dev: pd.DataFrame, protocol: str, fold: int, train: np.ndarray,
             block: np.ndarray) -> dict:
    t, v = dev.iloc[train], dev.iloc[block]
    later = int((t.feature_date > v.feature_date.min()).sum())
    return {
        "protocol": protocol, "fold": fold,
        "train_rows": len(train), "validation_rows": len(block),
        "train_feature_min": t.feature_date.min().date(),
        "train_feature_max": t.feature_date.max().date(),
        "train_label_max": t.label_date.max().date(),
        "validation_feature_min": v.feature_date.min().date(),
        "validation_feature_max": v.feature_date.max().date(),
        "validation_label_min": v.label_date.min().date(),
        "validation_label_max": v.label_date.max().date(),
        "training_rows_after_validation_start": later,
        "membership_sha256": hashlib.sha256(train.tobytes()).hexdigest()[:16],
    }


def main() -> int:
    rows = supervised_rows()
    dev = development(rows)
    blocks = validation_blocks(dev)
    plan, audit, failures = [], [], 0

    audit.append(f"supervised rows in cache      : {len(rows)}")
    audit.append(f"development rows by label_date: {len(dev)} "
                 f"(preregistered expectation {CONFIG['expected_development_rows']})")
    audit.append(f"development feature dates     : {dev.feature_date.min().date()} to {dev.feature_date.max().date()}")
    audit.append(f"development label dates       : {dev.label_date.min().date()} to {dev.label_date.max().date()}")
    if len(dev) != CONFIG["expected_development_rows"]:
        failures += 1
        audit.append("FAIL: development row count differs from the preregistered expectation; "
                     "the preregistration's stop rule applies (report, do not silently re-cut).")
    audit.append("")

    for fold, block in enumerate(blocks, start=1):
        ordered = ordered_window(dev, block)
        rnd = random_window(dev, block, fold)
        plan.append(describe(dev, "ordered_rolling_origin", fold, ordered, block))
        plan.append(describe(dev, "random_size_matched", fold, rnd, block))

        o, v = dev.iloc[ordered], dev.iloc[block]
        checks = [
            ("ordered: max train label_date < min validation feature_date",
             o.label_date.max() < v.feature_date.min()),
            ("ordered: max train feature_date < min validation feature_date",
             o.feature_date.max() < v.feature_date.min()),
            ("ordered: no training row inside the validation block",
             len(set(ordered) & set(block.tolist())) == 0),
            ("ordered: exactly the declared training size",
             len(ordered) == CONFIG["train_rows_per_fold"]),
            ("random: no training row inside the validation block",
             len(set(rnd.tolist()) & set(block.tolist())) == 0),
            ("random: purge rows before the block excluded",
             len(set(rnd.tolist()) & set(range(int(block[0]) - CONFIG["purge_rows_before_block"], int(block[0])))) == 0),
            ("random: embargo rows after the block excluded",
             len(set(rnd.tolist()) & set(range(int(block[-1]) + 1,
                 int(block[-1]) + 1 + CONFIG["embargo_rows_after_block"]))) == 0),
        ]
        audit.append(f"fold {fold}: validation {v.feature_date.min().date()} to {v.feature_date.max().date()} "
                     f"({len(block)} rows)")
        for label, ok in checks:
            audit.append(f"   {'pass' if ok else 'FAIL'}  {label}")
            failures += (not ok)
        after = int((dev.iloc[rnd].feature_date > v.feature_date.min()).sum())
        audit.append(f"   note  random arm draws {after} of {len(rnd)} training rows from after the "
                     f"validation block begins; this is the protocol difference under test")
        audit.append("")

    frame = pd.DataFrame(plan)
    frame.to_csv(HERE / "fold_plan.csv", index=False)
    audit.append(f"{'FAILED' if failures else 'All boundary invariants hold'}"
                 f"{'' if not failures else f': {failures} check(s)'}")
    audit.append("No model was fitted and no score was computed by this script.")
    text = "\n".join(audit)
    (HERE / "fold_plan_audit.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print("\nwrote fold_plan.csv and fold_plan_audit.txt")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
