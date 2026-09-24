from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import datetime as dt
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Fold:
    scheme: str
    number: int
    train: np.ndarray
    test: np.ndarray


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _utc_timestamp(date_text: str) -> int:
    return int(dt.datetime.fromisoformat(date_text).replace(tzinfo=dt.timezone.utc).timestamp())


def yahoo_url(config: dict) -> str:
    symbol = config["symbol"].split(".")[0]
    period1 = _utc_timestamp(config["start_date"])
    end_exclusive = (dt.date.fromisoformat(config["end_date"]) + dt.timedelta(days=1)).isoformat()
    period2 = _utc_timestamp(end_exclusive)
    return (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )


def download_yahoo(config: dict, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "aapl_yahoo_chart.json"
    url = yahoo_url(config)
    if refresh or not cache.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 FinanceMeta-repro/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        if b'"chart":{"result"' not in payload[:100]:
            raise RuntimeError("Yahoo did not return the expected chart JSON")
        cache.write_bytes(payload)
    payload = cache.read_bytes()
    document = json.loads(payload)
    result = document["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    adjusted_close = np.asarray(result["indicators"]["adjclose"][0]["adjclose"], dtype=float)
    raw_close = np.asarray(quote["close"], dtype=float)
    adjustment = adjusted_close / raw_close
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None),
            "Open": np.asarray(quote["open"], dtype=float) * adjustment,
            "High": np.asarray(quote["high"], dtype=float) * adjustment,
            "Low": np.asarray(quote["low"], dtype=float) * adjustment,
            "Close": adjusted_close,
            "Volume": quote["volume"],
        }
    )
    metadata = {
        "kind": "market",
        "provider": "Yahoo Finance chart endpoint",
        "url": url,
        "cache": str(cache.relative_to(ROOT)),
        "sha256": sha256_bytes(payload),
    }
    return frame, metadata


def synthetic_ohlcv(config: dict) -> tuple[pd.DataFrame, dict]:
    """Deterministic regime-shift data for code-path testing only."""
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(config["start_date"], config["end_date"])
    n = len(dates)
    returns = np.zeros(n)
    noise = rng.normal(0.0, 0.012, size=n)
    break_1, break_2 = int(n * 0.45), int(n * 0.78)
    for i in range(1, n):
        phi = 0.22 if i < break_1 else (-0.18 if i < break_2 else 0.0)
        returns[i] = phi * returns[i - 1] + noise[i]
    close = 30.0 * np.exp(np.cumsum(returns))
    overnight = rng.normal(0.0, 0.003, n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(overnight)
    spread = np.abs(rng.normal(0.009, 0.004, n))
    high = np.maximum(open_, close) * np.exp(spread / 2)
    low = np.minimum(open_, close) * np.exp(-spread / 2)
    volume = rng.lognormal(mean=17.5, sigma=0.35, size=n).astype(np.int64)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )
    payload = frame.to_csv(index=False).encode("utf-8")
    return frame, {
        "kind": "synthetic_smoke_test",
        "provider": "deterministic_internal_generator",
        "sha256": sha256_bytes(payload),
        "warning": "Synthetic output cannot answer the empirical equity-return question.",
    }


def make_dataset(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    data = raw.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data = data.sort_values("Date").drop_duplicates("Date").set_index("Date")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be strictly positive")

    log_close = np.log(data["Close"])
    daily_return = log_close.diff()
    for horizon in [1, 2, 3, 5, 10, 20]:
        data[f"ret_{horizon}"] = log_close.diff(horizon)
    data["volatility_5"] = daily_return.rolling(5).std(ddof=1)
    data["volatility_20"] = daily_return.rolling(20).std(ddof=1)
    data["intraday_range"] = np.log(data["High"] / data["Low"])
    data["overnight_return"] = np.log(data["Open"] / data["Close"].shift(1))
    log_volume = np.log1p(data["Volume"])
    volume_mean = log_volume.rolling(20).mean()
    volume_std = log_volume.rolling(20).std(ddof=1)
    data["volume_z20"] = (log_volume - volume_mean) / volume_std
    data["target"] = daily_return.shift(-1)

    columns = list(config["feature_columns"]) + ["target"]
    dataset = data[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if dataset.empty:
        raise ValueError("Feature construction produced an empty dataset")
    return dataset


def random_folds(n_rows: int, config: dict) -> list[Fold]:
    splitter = KFold(
        n_splits=int(config["random_folds"]),
        shuffle=True,
        random_state=int(config["random_seed"]),
    )
    return [Fold("random", i + 1, train, test) for i, (train, test) in enumerate(splitter.split(np.arange(n_rows)))]


def purged_walk_forward_folds(n_rows: int, config: dict) -> list[Fold]:
    n_folds = int(config["walk_forward_folds"])
    initial = int(np.floor(n_rows * float(config["walk_forward_initial_fraction"])))
    purge = int(config["purge_rows"])
    embargo = int(config["embargo_rows"])
    if initial <= purge:
        raise ValueError("Initial training window is too short for the purge")

    usable_for_tests = n_rows - initial - (n_folds - 1) * embargo
    block = usable_for_tests // n_folds
    if block < 1:
        raise ValueError("Not enough rows for the requested walk-forward folds")

    folds: list[Fold] = []
    embargoed: set[int] = set()
    cursor = initial
    for number in range(1, n_folds + 1):
        test_start = cursor
        test_end = n_rows if number == n_folds else test_start + block
        train_stop = test_start - purge
        train = np.array([i for i in range(train_stop) if i not in embargoed], dtype=int)
        test = np.arange(test_start, test_end, dtype=int)
        if train.size == 0 or test.size == 0:
            raise AssertionError("Every fold must have non-empty train and test sets")
        if train.max() >= test.min():
            raise AssertionError("Walk-forward training must be strictly earlier than validation")
        purged = set(range(train_stop, test_start))
        if purged.intersection(train) or purged.intersection(test):
            raise AssertionError("Purged rows leaked into a fold")
        if embargoed.intersection(train) or embargoed.intersection(test):
            raise AssertionError("Embargoed rows leaked into a later fold")
        folds.append(Fold("purged_walk_forward", number, train, test))
        if number < n_folds:
            embargo_range = range(test_end, min(test_end + embargo, n_rows))
            embargoed.update(embargo_range)
            cursor = test_end + embargo
    return folds


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def oos_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum(np.square(y)))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.sum(np.square(y - prediction))) / denominator


def pearson_ic(y: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(y) == 0.0 or np.std(prediction) == 0.0:
        return float("nan")
    return float(np.corrcoef(y, prediction)[0, 1])


def metrics(y: np.ndarray, prediction: np.ndarray, cost_bps: float) -> dict:
    position = np.sign(prediction)
    prior = np.r_[0.0, position[:-1]]
    turnover = np.abs(position - prior)
    net = position * y - (cost_bps / 10_000.0) * turnover
    net_std = float(np.std(net, ddof=1))
    net_sharpe = float(np.sqrt(252.0) * np.mean(net) / net_std) if net_std > 0 else float("nan")
    return {
        "oos_r2": oos_r2(y, prediction),
        "rmse": float(np.sqrt(np.mean(np.square(y - prediction)))),
        "information_coefficient": pearson_ic(y, prediction),
        "sign_accuracy": float(np.mean(np.sign(y) == np.sign(prediction))),
        "net_sharpe_5bps": net_sharpe,
    }


def evaluate_grid(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    folds: Iterable[Fold],
    alphas: list[float],
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = list(folds)
    score_rows: list[dict] = []
    boundary_rows: list[dict] = []

    for fold in folds:
        train_dates = dates[fold.train]
        test_dates = dates[fold.test]
        boundary_rows.append(
            {
                "scheme": fold.scheme,
                "fold": fold.number,
                "train_rows": len(fold.train),
                "test_rows": len(fold.test),
                "train_min": train_dates.min().date().isoformat(),
                "train_max": train_dates.max().date().isoformat(),
                "test_min": test_dates.min().date().isoformat(),
                "test_max": test_dates.max().date().isoformat(),
                "training_rows_after_first_test_date": int(np.sum(train_dates > test_dates.min())),
            }
        )

    for alpha in alphas:
        pooled_y: list[np.ndarray] = []
        pooled_prediction: list[np.ndarray] = []
        fold_metrics: list[dict] = []
        for fold in folds:
            model = make_pipeline(alpha)
            model.fit(X[fold.train], y[fold.train])
            prediction = model.predict(X[fold.test])
            fold_result = metrics(y[fold.test], prediction, cost_bps)
            fold_result["fold"] = fold.number
            fold_metrics.append(fold_result)
            pooled_y.append(y[fold.test])
            pooled_prediction.append(prediction)
        pooled = metrics(np.concatenate(pooled_y), np.concatenate(pooled_prediction), cost_bps)
        score_rows.append(
            {
                "scheme": folds[0].scheme,
                "alpha": alpha,
                "mean_fold_oos_r2": float(np.mean([row["oos_r2"] for row in fold_metrics])),
                "std_fold_oos_r2": float(np.std([row["oos_r2"] for row in fold_metrics], ddof=1)),
                **{f"pooled_{key}": value for key, value in pooled.items()},
            }
        )
    return pd.DataFrame(score_rows), pd.DataFrame(boundary_rows)


def choose_alpha(scores: pd.DataFrame) -> pd.Series:
    # Stable, predeclared tie-break: larger alpha wins.
    return scores.sort_values(["mean_fold_oos_r2", "alpha"], ascending=[False, False]).iloc[0]


def final_fit_and_score(
    development: pd.DataFrame,
    lockbox: pd.DataFrame,
    features: list[str],
    alpha: float,
    cost_bps: float,
    purge_rows: int,
) -> tuple[dict, np.ndarray]:
    eligible = development.iloc[: len(development) - purge_rows] if purge_rows else development
    model = make_pipeline(alpha)
    model.fit(eligible[features].to_numpy(), eligible["target"].to_numpy())
    prediction = model.predict(lockbox[features].to_numpy())
    return metrics(lockbox["target"].to_numpy(), prediction, cost_bps), prediction


def write_plot(summary: dict) -> None:
    schemes = ["random", "purged_walk_forward"]
    validation = [summary["schemes"][name]["validation_mean_fold_oos_r2"] for name in schemes]
    lockbox = [summary["schemes"][name]["lockbox"]["oos_r2"] for name in schemes]
    x = np.arange(len(schemes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, validation, width, label="Validation")
    ax.bar(x + width / 2, lockbox, width, label="2024–2025 lockbox")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, ["Random", "Purged walk-forward"])
    ax.set_ylabel("Zero-benchmark OOS R²")
    ax.set_title("Same features and Ridge model; only validation changes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    config = load_config()
    config_hash = sha256_file(CONFIG_PATH)
    raw, data_metadata = (
        synthetic_ohlcv(config) if args.synthetic_smoke_test else download_yahoo(config, args.refresh_data)
    )
    dataset = make_dataset(raw, config)
    development = dataset.loc[: config["development_end"]].copy()
    lockbox = dataset.loc[config["lockbox_start"] : config["end_date"]].copy()
    if development.empty or lockbox.empty:
        raise ValueError("Development and lockbox samples must both be non-empty")
    if development.index.max() >= lockbox.index.min():
        raise AssertionError("Development and lockbox dates overlap")

    features = list(config["feature_columns"])
    X_dev = development[features].to_numpy()
    y_dev = development["target"].to_numpy()
    dates = development.index
    cost_bps = float(config["one_way_transaction_cost_bps"])
    alphas = [float(value) for value in config["ridge_alphas"]]

    random_split = random_folds(len(development), config)
    safe_split = purged_walk_forward_folds(len(development), config)
    random_scores, random_boundaries = evaluate_grid(X_dev, y_dev, dates, random_split, alphas, cost_bps)
    safe_scores, safe_boundaries = evaluate_grid(X_dev, y_dev, dates, safe_split, alphas, cost_bps)
    scores = pd.concat([random_scores, safe_scores], ignore_index=True)
    boundaries = pd.concat([random_boundaries, safe_boundaries], ignore_index=True)

    selected_random = choose_alpha(random_scores)
    selected_safe = choose_alpha(safe_scores)
    random_final, random_prediction = final_fit_and_score(
        development, lockbox, features, float(selected_random["alpha"]), cost_bps, purge_rows=0
    )
    safe_final, safe_prediction = final_fit_and_score(
        development,
        lockbox,
        features,
        float(selected_safe["alpha"]),
        cost_bps,
        purge_rows=int(config["purge_rows"]),
    )

    random_val = float(selected_random["mean_fold_oos_r2"])
    safe_val = float(selected_safe["mean_fold_oos_r2"])
    random_optimism = random_val - float(random_final["oos_r2"])
    safe_optimism = safe_val - float(safe_final["oos_r2"])
    minimum_effect = float(config["minimum_effect"])
    condition_a = random_val - safe_val >= minimum_effect
    condition_b = random_optimism - safe_optimism >= minimum_effect
    supported = bool(condition_a and condition_b)

    summary = {
        "status": "synthetic_smoke_test_only" if args.synthetic_smoke_test else "empirical_result",
        "config_sha256": config_hash,
        "data": data_metadata,
        "sample": {
            "development_rows": len(development),
            "development_start": development.index.min().date().isoformat(),
            "development_end": development.index.max().date().isoformat(),
            "lockbox_rows": len(lockbox),
            "lockbox_start": lockbox.index.min().date().isoformat(),
            "lockbox_end": lockbox.index.max().date().isoformat(),
        },
        "schemes": {
            "random": {
                "selected_alpha": float(selected_random["alpha"]),
                "validation_mean_fold_oos_r2": random_val,
                "validation_pooled_oos_r2": float(selected_random["pooled_oos_r2"]),
                "lockbox": random_final,
                "validation_optimism": random_optimism,
            },
            "purged_walk_forward": {
                "selected_alpha": float(selected_safe["alpha"]),
                "validation_mean_fold_oos_r2": safe_val,
                "validation_pooled_oos_r2": float(selected_safe["pooled_oos_r2"]),
                "lockbox": safe_final,
                "validation_optimism": safe_optimism,
            },
        },
        "decision_rule": {
            "minimum_effect": minimum_effect,
            "condition_a_random_minus_safe_validation": random_val - safe_val,
            "condition_a_pass": bool(condition_a),
            "condition_b_random_minus_safe_optimism": random_optimism - safe_optimism,
            "condition_b_pass": bool(condition_b),
            "hypothesis_supported": supported,
            "conclusion": (
                "Random splitting is materially optimistic under the predeclared rule."
                if supported
                else "The predeclared hypothesis failed; preserve this result without retuning."
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "validation_scores.csv", index=False)
    boundaries.to_csv(RESULTS_DIR / "fold_boundaries.csv", index=False)
    predictions = pd.DataFrame(
        {
            "date": lockbox.index,
            "actual_next_day_log_return": lockbox["target"].to_numpy(),
            "random_selected_prediction": random_prediction,
            "safe_selected_prediction": safe_prediction,
        }
    )
    predictions.to_csv(RESULTS_DIR / "lockbox_predictions.csv", index=False)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_plot(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help="Verify the pipeline on deterministic synthetic data; not empirical evidence.",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Redownload the frozen Stooq date range.")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["decision_rule"], indent=2))
